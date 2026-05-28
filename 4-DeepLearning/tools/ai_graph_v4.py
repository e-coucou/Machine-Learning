import json
import math
import os
from datetime import datetime, timedelta
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ===========================================================================
# TRAINING CONFIG  —  constantes globales du projet
# ===========================================================================

class TrainingConfig:
    """Constantes globales de l'entraînement et chemins de fichiers."""

    VERSION      = 'v4.0.2'
    MONITOR_FILE = 'model/monitor.log'
    TRAIN_INFO   = 'model/train_info.json'

    TARGET_BLOCK_DS1 = 2525599   # 3367466 / 5051199 / old wiki_raw 5963390
    TARGET_BLOCK_DS2 = 1203058   # 1619593 / 2429390
    TARGET_BLOCK_DS3 = 85440     # 116091  / 174137

    MIXED      = 0.05
    LITT       = 0.05
    BATCH_SIZE = 4
    GRAD_ACCUM = 96
    END_TRAIN  = 30000

    # Configuration des phases d'entraînement pour le monitoring (30 000 steps)
    # Format : (start, stop, titre, couleur, nom_technique, etat_lr, neurones_action, symptome)
    PHASES_TRAIN = [
        (
            0, 1500,
            "Éveil & Explosion",
            "#cc3333",                        # Rouge clair (Chaud / Instable)
            "Warmup & Initial Convergence",
            "Ascension rapide : 0 \u2192 3.5e-4",
            "Découverte du bruit. Apprentissage des espaces et des mots très fréquents (le, la, et).",
            "Charabia total, puis mots existants mais sans suite logique."
        ),
        (
            1500, 5000,
            "Acquisition Syntaxique",
            "#ffbc59",                        # Orange clair (Forge / Grammaire)
            "Grammar & Structural Learning",
            "Plateau maximum : ~3.5e-4",
            "Intégration de la syntaxe. Mémorisation de la ponctuation. Associations statistiques par proximité pure.",
            "Le Poète Amnésique : Grammaire parfaite, style riche, mais hallucinations géographiques/logiques."
        ),
        (
            5000, 12000,
            "Ségrégation Sémantique",
            "#c3c311",                        # Jaune (Tri / Frontières)
            "Domain Separation & Knowledge Embedding",
            "Descente douce : 3.4e-4 \u2192 2.5e-4",
            "Création de cloisons dans l'espace latent. Séparation des données Wiki (faits) et CulturaX (web/produits).",
            "Les faits se précisent. Baisse drastique des associations absurdes (couleurs = musée)."
        ),
        (
            12000, 22000,
            "Cristallisation",
            "#79df79",                        # Vert clair (Solidification / Logique)
            "Attention Head Specialization",
            "Chute raide : 2.5e-4 \u2192 < 1.0e-4",
            "Figeage des poids. Les têtes d'attention se spécialisent (une pour la géographie, une pour la logique).",
            "Apparition du raisonnement (Bleu + Jaune = Vert). Fin des boucles de répétition."
        ),
        (
            22000, 30000,
            "Affinage & Polissage",
            "#89acef",                        # Bleu clair (Froid / Précision)
            "Late-stage Fine-Tuning & Entropy Reduction",
            "Atterrissage lent : < 1.0e-4 \u2192 3.5e-5",
            "Correction des micro-détails et du vocabulaire rare. Baisse de l'entropie (incertitude).",
            "Réponses stables, déterministes, nettes et précises."
        )
    ]


# ===========================================================================
# MATH UTILS  —  calculs purs, sans état
# ===========================================================================

class MathUtils:
    """Fonctions mathématiques et calculs ML (stateless)."""

    @staticmethod
    def calcul_ds(step, mix, grad):
        return int(step * mix * grad)

    @staticmethod
    def calcul_epoch(step, target, mix, batch, grad, t_step, n=1):
        epoch, next_block = 0, 0
        if mix > 0:
            n_blocks   = (target * n) - (step * batch)
            epoch      = int((n_blocks) / (batch * mix * grad) + t_step)
            next_block = n_blocks / 4 + step
        return epoch, next_block

    @staticmethod
    def get_epoch(first, target, mix, batch, grad, t_step):
        epochs = []
        n      = int(first * batch / target) + 1
        e, t   = MathUtils.calcul_epoch(
            step=first, target=target, mix=mix, batch=batch, grad=grad, t_step=t_step, n=n
        )
        epochs.append(e)
        while e < TrainingConfig.END_TRAIN:
            n   += 1
            e, t = MathUtils.calcul_epoch(
                step=t, target=target, mix=mix, batch=batch, grad=grad, t_step=e, n=n
            )
            if e < TrainingConfig.END_TRAIN:
                epochs.append(e)
        return epochs

    @staticmethod
    def calcul_ema(data_loss, data_steps, compare=None):
        if len(data_loss) == 0:
            return 0, 0
        ema_loss = data_loss[0]
        if compare is not None:
            try:
                end = np.where(data_steps >= compare)[0][0]
            except IndexError:
                end = len(data_steps) - 1
        else:
            end = len(data_steps) - 1
        for l in data_loss[:end + 1]:
            ema_loss = 0.1 * l + (1 - 0.1) * ema_loss
        return ema_loss, end

    @staticmethod
    def calcul_lr(it):
        warmup    = 1500
        max_iters = 30000
        lr_max    = 0.00035
        lr_min    = lr_max * 0.1
        # 1) Phase de warmup
        if it < warmup:
            return lr_max * it / warmup
        # 2) Phase de plateau bas
        if it > max_iters:
            return lr_min
        # 3) Phase de Cosine Decay
        decay_ratio = (it - warmup) / (max_iters - warmup)
        coeff       = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return lr_min + coeff * (lr_max - lr_min)

    @staticmethod
    def moving_average(x, w):
        if w <= 1:
            return x
        return np.convolve(x, np.ones(w), "valid") / w

    @staticmethod
    def predict_loss_projection(steps, losses, target_step, start_idx=-1):
        """
        Projection robuste de la Loss pour LLM.
        On bloque les coefficients dans une zone 'réelle'.
        Modèle : L(s) = a * s^-b + c
        """
        def power_law(s, a, b, c):
            return a * np.power(s, -b) + c

        if start_idx == -1:
            start_idx = len(steps) // 10
        fit_steps  = np.array(steps[start_idx:])
        fit_losses = np.array(losses[start_idx:])

        try:
            # Bornes : a > 0, b vitesse [0.01–0.8], c asymptote [1.5–4.0]
            lower_bounds = [0.1, 0.01, 1.5]
            upper_bounds = [1000, 0.8, 4.0]
            # Poids dégressifs : plus d'importance aux points récents
            sigmas = np.ones(len(fit_steps))
            sigmas[-max(1, len(sigmas) // 10):] = 0.5
            p0 = [10, 0.2, 2.8]
            popt, _ = curve_fit(
                power_law, fit_steps, fit_losses,
                p0=p0, bounds=(lower_bounds, upper_bounds),
                sigma=sigmas, maxfev=20000
            )
            future_steps = np.arange(steps[-1], target_step, 100)
            projection   = power_law(future_steps, *popt)
            return future_steps, projection, popt
        except Exception as e:
            print(f"Erreur d'extrapolation : {e}")
            return None, None, None

    @staticmethod
    def predict_loss_projection_old(steps, losses, target_step, start_idx=-1):
        """
        ATTENTION : version un peu instable.
        Projette la perte (Loss) vers un step futur.
        steps   : liste des steps actuels (ex: [100, 200, ... 4100])
        losses  : liste des valeurs de loss correspondantes
        target_step : le step jusqu'auquel on veut voir le futur (ex: 15000)
        """
        def power_law(s, a, b, c):
            return a * np.power(s, -b) + c

        if start_idx == -1:
            start_idx = len(steps) // 5
        fit_steps  = np.array(steps[start_idx:])
        fit_losses = np.array(losses[start_idx:])

        try:
            p0 = [10, 0.5, 2.0]
            popt, _ = curve_fit(
                power_law, fit_steps, fit_losses, p0=p0,
                bounds=([0, 0.01, 1.0], [np.inf, 1.0, 4.0])
            )
            future_steps = np.arange(steps[-1], target_step, 100)
            projection   = power_law(future_steps, *popt)
            return future_steps, projection, popt
        except Exception as e:
            print(f"Erreur d'extrapolation : {e}")
            return None, None, None

    @staticmethod
    def check_efficiency(popt, current_step, target_step=60000):
        """Analyse l'efficacité de l'apprentissage à un horizon donné."""
        a, b, c = popt
        # 1. Perte projetée à l'horizon  L(s) = a * s^-b + c
        loss_at_horizon = a * np.power(target_step, -b) + c
        # 2. Pente au step actuel  d/ds [a*s^-b] = -a * b * s^-(b+1)
        current_slope   = -a * b * np.power(current_step, -(b + 1))
        # 3. Pente à l'horizon
        future_slope    = -a * b * np.power(target_step, -(b + 1))
        return {
            "loss_horizon":           round(loss_at_horizon, 4),
            "current_slope_per_1000": round(current_slope * 1000, 5),   # gain / 1 000 steps
            "future_slope_per_1000":  round(future_slope  * 1000, 5),
            "is_plateau":             abs(future_slope * 1000) < 0.005  # seuil de stagnation
        }


# ===========================================================================
# DATA LOADER  —  chargement et parsing des sources de données
# ===========================================================================

class DataLoader:
    """Charge et prépare toutes les sources de données (JSON logs, monitor, train_info)."""

    def __init__(self, config: TrainingConfig = None):
        self.cfg  = config or TrainingConfig()
        self.info = None   # contenu de train_info.json

    # ---- Méta-info --------------------------------------------------------

    def load_train_info(self, path=None):
        path = path or self.cfg.TRAIN_INFO
        with open(path, 'r') as f:
            self.info = json.load(f)
        return self.info

    # ---- Conversions steps → tokens ---------------------------------------

    @staticmethod
    def step_to_tokens(step, model_name, config):
        """Convertit un step spécifique d'un modèle en nombre de tokens total."""
        if model_name not in config:
            raise ValueError(f"Modèle {model_name} non trouvé dans le JSON")
        phases       = config[model_name]
        total_tokens = 0
        for phase in phases:
            start = phase['start_step']
            end   = phase['end_step']
            if step > start:
                steps_in_phase = min(step, end) - start
                total_tokens  += steps_in_phase * phase['ebs'] * phase['block_size']
        return total_tokens

    def convert_steps_list(self, steps_list, model_name):
        """Macro : convertit une liste de steps en millions de tokens."""
        return [
            DataLoader.step_to_tokens(s, model_name, self.info) / 1_000_000
            for s in steps_list
        ]

    # ---- Fichiers de log JSON ---------------------------------------------

    def load_main(self, log_path):
        if not os.path.exists(log_path):
            print(f"❌ Fichier principal introuvable : {log_path}")
            return None
        with open(log_path, "r") as f:
            return json.load(f)

    def load_second(self, second):
        if second and os.path.exists(second):
            with open(second, "r") as f:
                return json.load(f)
        if not second:
            print(f"⚠️  Note : Second fichier de log non chargé ({second})")
        return None

    def load_third(self, third):
        if third and os.path.exists(third):
            with open(third, "r") as f:
                return json.load(f)
        if not third:
            print(f"⚠️  Note : Troisième fichier de log non chargé ({third})")
        return None

    # ---- Monitor JSONL ----------------------------------------------------

    def load_monitor(self, monitor_file=None):
        """Charge et parse le fichier monitor.log (JSONL) — insère un point à 0."""
        path = monitor_file or self.cfg.MONITOR_FILE
        cfg  = self.cfg

        data_monitor = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data_monitor.append(json.loads(line))

        # Listes pré-initialisées avec le point zéro
        monitor_loss    = [0]
        monitor_lr      = [0]
        monitor_ds1     = [0]
        monitor_ds2     = [0]
        monitor_ds3     = [0]
        monitor_mixed   = [0]
        monitor_step_ds = [0]
        monitor_grad    = [0]
        monitor_token   = [0]

        for d in data_monitor:
            monitor_loss.append(d['loss'])
            monitor_step_ds.append(d['step'])
            monitor_lr.append(float(d['lr']) * 1.e4)
            monitor_grad.append(np.mean(d['grad_norm']))
            monitor_ds1.append(
                (d.get('dataset1', 0) / cfg.TARGET_BLOCK_DS1 * cfg.BATCH_SIZE) % 1)
            monitor_ds2.append(
                (d.get('dataset2', 0) / cfg.TARGET_BLOCK_DS2 * cfg.BATCH_SIZE) % 1)
            monitor_ds3.append(
                (d.get('dataset3', 0) / cfg.TARGET_BLOCK_DS3 * cfg.BATCH_SIZE) % 1)
            monitor_mixed.append(
                d.get('dataset1', 0) / (
                    d.get('dataset1', 0) + d.get('dataset2', 0.1) + d.get('dataset3', 0.1)
                )
            )
            monitor_token.append(
                int(d['step'] * d['params']['batch_size']
                    * d['params']['grad_accum_steps'] / 100_000) * 100
            )

        return {
            "raw":   data_monitor,
            "loss":  monitor_loss,
            "lr":    monitor_lr,
            "ds1":   monitor_ds1,
            "ds2":   monitor_ds2,
            "ds3":   monitor_ds3,
            "mixed": monitor_mixed,
            "steps": monitor_step_ds,
            "grad":  monitor_grad,
            "token": monitor_token,
        }


# ===========================================================================
# PLOT UTILS  —  annotations, phases, copyright
# ===========================================================================

class PlotUtils:
    """Utilitaires graphiques : labels, curseur, phases d'entraînement, copyright."""

    @staticmethod
    def add_label(ax, label, value, offset, left=1., color="gray", textAlign='left'):
        ax.annotate(
            label,
            xy=(left, value),
            xytext=(offset, 0),
            xycoords=('axes fraction', 'data'),
            textcoords='offset points',
            fontweight='bold', fontsize=8, color=color,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='square,pad=0'),
            va='center', ha=textAlign
        )

    @staticmethod
    def add_cursor_x(ax, label, value, ymin, offset, color="gray", textAlign='up'):
        ax.annotate(
            label,
            xy=(value, 0),
            xytext=(0, -offset),
            xycoords=('data', 'axes fraction'),
            textcoords='offset points',
            fontweight='bold', fontsize=8, color=color,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='square,pad=0'),
            ha='center', va=textAlign
        )

    @staticmethod
    def set_off(ligne, legende):
        ligne.set_visible(False)
        legende.set_alpha(0.2)

    @staticmethod
    def ajouter_phase_train(ax, start, end, phases):
        for debut, fin, label, color, _, _, _, _ in phases:
            if debut < end:
                ax.axvspan(debut, end, ymin=0, ymax=0.02,
                           facecolor=color, alpha=1, edgecolor='gray', linewidth=0.1)
                xText = (max(debut, start) + min(fin, end)) // 2
                ax.text(xText, 0.01, label,
                        transform=ax.get_xaxis_transform(),
                        ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')

    @staticmethod
    def get_phase_train(step, phases):
        for phase in phases:
            start, stop, titre, couleur, tech, lr, neurone, symptome = phase
            if start < step < stop or (step == stop and step == phases[-1][1]):
                return {
                    "start": start, "stop": stop, "titre": titre,
                    "couleur": couleur, "tech": tech, "lr": lr,
                    "neurone": neurone, "symptome": symptome
                }

    @staticmethod
    def copyright(ax, ymin):
        label = f'eCoucou {TrainingConfig.VERSION}'
        print(f"-" * 100, "\n", label, "-")
        ax.annotate(
            label,
            xy=(1, 0), xytext=(50, -15),
            xycoords=('axes fraction', 'axes fraction'),
            textcoords='offset points',
            fontsize=6, color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='square,pad=0'),
            va='top', ha='center'
        )


# ===========================================================================
# POUSSIN PLOTTER  —  classe principale de rendu
# ===========================================================================

class PoussinPlotter:
    """
    Moteur de rendu du graphique d'entraînement Poussin.
    Encapsule la figure Matplotlib, les événements interactifs
    et toute la logique de tracé.
    """

    def __init__(
        self,
        y_min=2.6,
        y_max=3.3,
        x_max=70000,
        x_min=0,
        smooth=5,
        log_path="model/my_wiky_history.json",
        second="save/model_3/my_wiky_history.json",
        third=None,
        compare=None,
        annot_event=None,
        target=None,
        speed=None,
        raw=False,
        titre="Training GPT Mac-M1",
        horizon=30000,
        proj=10,
        phase_train=None,
        ds=False,
        config=None,
    ):
        self.cfg         = config or TrainingConfig()
        self.y_min       = y_min
        self.y_max       = y_max
        self.x_max       = x_max
        self.x_min       = x_min
        self.smooth      = smooth
        self.log_path    = log_path
        self.second      = second
        self.third       = third
        self.compare     = compare
        self.annot_event = annot_event
        self.target      = target
        self.speed       = speed
        self.raw         = raw
        self.titre       = titre
        self.horizon     = horizon
        self.proj        = proj
        self.phase_train = phase_train if phase_train is not None else self.cfg.PHASES_TRAIN
        self.ds          = ds

        # État interactif (remplace les anciens globals)
        self.xActual    = 0
        self.spdActual  = 0
        self.line_dict  = {}
        self.trend      = []
        self.trend_line = []
        self.fig        = None
        self.ax         = None

    # ------------------------------------------------------------------
    # Gestionnaires d'événements Matplotlib
    # ------------------------------------------------------------------

    def on_mouse_move(self, event):
        if event.inaxes:
            v_line = event.canvas.figure.v_line
            h_line = event.canvas.figure.h_line
            t_line = event.canvas.figure.t_line
            v_line.set_xdata([event.xdata, event.xdata])
            v_line.set_visible(True)
            ax = h_line.axes
            x, y = ax.transData.inverted().transform((event.x, event.y))
            h_line.set_ydata([y, y])
            h_line.set_visible(True)
            ax3     = t_line.axes
            _, y3   = ax3.transData.inverted().transform((event.x, event.y + 20))

            if (x > self.xActual) and (event.y > 101):
                lr     = MathUtils.calcul_lr(x)
                elapse = (x - self.xActual) * self.spdActual
                eta    = datetime.now() + timedelta(seconds=elapse)
                ETA    = eta.strftime("%d-%B %H:%M")
                t_line.set_text(
                    f"{ETA}\nlr = {lr:10.8f}\n[{round(x):5d} - {y:4.2f}]"
                )
                t_line.get_bbox_patch().set_facecolor('#ccddff')
                t_line.set_ha('center')
                t_line.xy = (x, y3)
                t_line.set_visible(True)
            elif event.y < 100:
                info = PlotUtils.get_phase_train(round(x), self.phase_train)
                t_line.set_text(
                    f"{info['titre']}  [{info['start']}-{info['stop']}]\n"
                    f"{info['tech']}\n{info['lr']}\n{info['neurone']}\n"
                    f"{info['symptome']}\neCoucou {self.cfg.VERSION}"
                )
                t_line.xy = (x, y3 + 0.2)
                t_line.get_bbox_patch().set_facecolor(info['couleur'])
                t_line.set_ha('left')
                t_line.set_zorder(1000)
                t_line.set_visible(True)
            else:
                t_line.set_visible(False)
            event.canvas.draw_idle()

    def on_mouse_click(self, event):
        if event.button == 3:
            for a in self.trend_line:
                a.remove()
            self.trend_line.clear()
            self.trend.clear()
            event.canvas.draw_idle()
            return

        if event.inaxes and event.button == 1:
            ax_p = event.canvas.figure.ax_principal
            inv  = ax_p.transData.inverted()
            x, y = inv.transform((event.x, event.y))
            self.trend.append((x, y))
            pt, = ax_p.plot(x, y, 'ro', markersize=2)
            self.trend_line.append(pt)

        if len(self.trend) == 2:
            p1, p2 = self.trend[0], self.trend[1]
            tl, = ax_p.plot(
                [p1[0], p2[0]], [p1[1], p2[1]],
                color="orange", linestyle="-", marker='', lw=2, zorder=1000
            )
            self.trend_line.append(tl)
            self.trend.clear()
            event.canvas.draw_idle()

    def on_pick(self, event):
        leg_item = event.artist
        if leg_item in self.line_dict:
            orig_line = self.line_dict[leg_item][0]
            vis       = not orig_line.get_visible()
            orig_line.set_visible(vis)
            leg_item.set_alpha(1. if vis else 0.2)
            event.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def plot(self):
        cfg = self.cfg
        mu  = MathUtils
        pu  = PlotUtils

        y_min  = self.y_min
        y_max  = self.y_max
        x_min  = self.x_min
        x_max  = self.x_max
        smooth = self.smooth
        raw    = self.raw
        ds     = self.ds
        x_sens = (x_max - x_min) * 0.013

        # ---- Chargement des données ----------------------------------------
        dl   = DataLoader(cfg)
        info = dl.load_train_info()

        data = dl.load_main(self.log_path)
        if data is None:
            return

        mon             = dl.load_monitor()
        data_monitor    = mon["raw"]
        monitor_loss    = mon["loss"]
        monitor_lr      = mon["lr"]
        monitor_ds1     = mon["ds1"]
        monitor_ds2     = mon["ds2"]
        monitor_ds3     = mon["ds3"]
        monitor_mixed   = mon["mixed"]
        monitor_step_ds = mon["steps"]
        monitor_grad    = mon["grad"]
        monitor_token   = mon["token"]
        monitor_tokens  = dl.convert_steps_list(monitor_step_ds, "model_5")

        compare = self.compare
        if compare == -1:
            compare = data["steps"][-1]

        epoch_1 = mu.get_epoch(
            first=data_monitor[-1]['dataset1'], target=cfg.TARGET_BLOCK_DS1,
            mix=(1 - cfg.MIXED - cfg.LITT), batch=cfg.BATCH_SIZE,
            grad=cfg.GRAD_ACCUM, t_step=compare
        )
        epoch_2 = mu.get_epoch(
            first=data_monitor[-1]['dataset2'], target=cfg.TARGET_BLOCK_DS2,
            mix=cfg.MIXED, batch=cfg.BATCH_SIZE,
            grad=cfg.GRAD_ACCUM, t_step=compare
        )
        epoch_3 = mu.get_epoch(
            first=data_monitor[-1]['dataset3'], target=cfg.TARGET_BLOCK_DS3,
            mix=cfg.LITT, batch=cfg.BATCH_SIZE,
            grad=cfg.GRAD_ACCUM, t_step=compare
        )

        steps           = np.array(data["steps"])
        train_loss      = np.array(data["train_loss"])
        val_loss        = np.array(data["val_loss"])
        val_loss_wiki   = np.array(data["val_loss_wiki"])
        val_loss_cult   = np.array(data["val_loss_cult"])
        val_loss_litt   = np.array(data["val_loss_litt"])
        train_loss_wiki = np.array(data["train_loss_wiki"])
        train_loss_cult = np.array(data["train_loss_cult"])
        train_loss_litt = np.array(data["train_loss_litt"])
        gap    = val_loss - train_loss
        tokens = dl.convert_steps_list(steps, "model_5")

        # ---- Données de comparaison (modèles 2 et 3) -----------------------
        data_second = dl.load_second(self.second)
        if data_second:
            val_loss_second  = np.array(data_second["val_loss"])
            steps_second     = np.array(data_second["steps"])
            tokens_second    = dl.convert_steps_list(steps_second, "model_5")
            ema_loss_second, end_second = mu.calcul_ema(val_loss_second, steps_second, compare=compare)
        else:
            val_loss_second = steps_second = None
            ema_loss_second = end_second = 0

        data_third = dl.load_third(self.third)
        if data_third:
            val_loss_third  = np.array(data_third["val_loss"])
            steps_third     = np.array(data_third["steps"])
            ema_loss_third, end_third = mu.calcul_ema(val_loss_third, steps_third, compare=compare)
        else:
            val_loss_third = steps_third = None
            ema_loss_third = end_third = 0

        # ---- EMA principales -----------------------------------------------
        ema_loss,  end      = mu.calcul_ema(val_loss, steps, compare=compare)
        ema_last,  end_last = mu.calcul_ema(val_loss, steps, compare=None)

        # ---- Style & figure ------------------------------------------------
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(15, 7.5))
        self.fig = fig
        self.ax  = ax
        fig.ax_principal = ax
        plt.subplots_adjust(left=0.03, right=0.88, top=0.92, bottom=0.06)

        self.xActual   = data_monitor[-1]['step']
        self.spdActual = data_monitor[-1]['elapse'] / (self.xActual - data_monitor[-2]['step'])

        fig.v_line = ax.axvline(color='#aaaaaa', linestyle='-.', linewidth=0.8,
                                visible=False, zorder=10)
        fig.h_line = ax.axhline(color='#aaaaaa', linestyle='-.', linewidth=0.8,
                                visible=False, zorder=10)

        fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        fig.canvas.mpl_connect('button_press_event',  self.on_mouse_click)

        # ---- Axes ticks X / Y ----------------------------------------------
        if (x_max - x_min) < 7001:
            x_ticks = 100
        elif (x_max - x_min) < 10001:
            x_ticks = 200
        else:
            x_ticks = 500
        epsilon = 1e-5
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_ticks))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
        ax.set_yticks(np.arange(y_min, y_max + epsilon, 0.1))
        ax.tick_params(axis='y', colors='orange',  labelsize=8, length=2, width=0.7)
        ax.tick_params(axis='x', which='major', length=5, width=1, colors='black', labelsize=8)
        ax.tick_params(axis='x', which='minor', length=2, width=0.5, colors='gray')
        plt.xticks(rotation=45)

        # ---- Axe X supérieur (tokens vus) ----------------------------------
        ay = ax.twiny()
        ay.grid(False)
        ay.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ay.xaxis.set_minor_locator(ticker.AutoMinorLocator(0.1))
        ay_min = int((x_min * cfg.BATCH_SIZE * cfg.GRAD_ACCUM) * 0.512 / 10_000) / 100
        ay_max = int((x_max * cfg.BATCH_SIZE * cfg.GRAD_ACCUM * 0.512) / 10_000) / 100
        ay.set_xlim(ay_min, ay_max)
        ay.set_xticks(np.arange(ay_min - epsilon, ay_max, 0.05))
        ay.tick_params(axis='x', labelsize=8, color='black', length=2, width=0.7, rotation=45)

        pu.add_cursor_x(ax, "\u25b2", data_monitor[-1]['step'], y_min, 2,
                        color="black", textAlign='top')

        # ---- Zones GAP (overfitting check) ---------------------------------
        for i in range(len(steps) - 1):
            g_val = gap[i]
            c     = "#acffaa" if g_val <= 0.05 else ("#ffff00" if g_val <= 0.12 else "red")
            ax.axvspan(steps[i], steps[i + 1], facecolor=c, alpha=0.1)

        # ---- Bandeau phases d'entraînement ---------------------------------
        pu.ajouter_phase_train(ax, x_min, x_max, self.phase_train)

        # ---- Calcul de toutes les courbes EMA ------------------------------
        ema_smooth          = []
        ema_smooth_raw      = []
        ema_smooth_2        = []
        ema_smooth_3        = []
        ema_val_loss_wiki   = []
        ema_val_loss_cult   = []
        ema_val_loss_litt   = []
        ema_train_loss_wiki = []
        ema_train_loss_cult = []
        ema_train_loss_litt = []
        ema_train_loss      = []

        for i in range(len(steps)):
            ema_smooth.append(         mu.calcul_ema(val_loss[:i+1],        steps[:i+1])[0])
            ema_val_loss_wiki.append(  mu.calcul_ema(val_loss_wiki[:i+1],   steps[:i+1])[0])
            ema_val_loss_cult.append(  mu.calcul_ema(val_loss_cult[:i+1],   steps[:i+1])[0])
            ema_val_loss_litt.append(  mu.calcul_ema(val_loss_litt[:i+1],   steps[:i+1])[0])
            ema_train_loss_wiki.append(mu.calcul_ema(train_loss_wiki[:i+1], steps[:i+1])[0])
            ema_train_loss_cult.append(mu.calcul_ema(train_loss_cult[:i+1], steps[:i+1])[0])
            ema_train_loss_litt.append(mu.calcul_ema(train_loss_litt[:i+1], steps[:i+1])[0])
            ema_train_loss.append(     mu.calcul_ema(train_loss[:i+1],      steps[:i+1])[0])

        if val_loss_second is not None:
            for i in range(len(steps_second)):
                ema_smooth_2.append(mu.calcul_ema(val_loss_second[:i+1], steps_second[:i+1])[0])

        if val_loss_third is not None:
            for i in range(len(steps_third)):
                ema_smooth_3.append(mu.calcul_ema(val_loss_third[:i+1], steps_third[:i+1])[0])

        if raw:
            for i in range(len(monitor_step_ds)):
                ema_smooth_raw.append(mu.calcul_ema(monitor_loss[:i+1], monitor_step_ds[:i+1])[0])

        # ---- Projection loi de puissance -----------------------------------
        steps_proj, ema_proj, popt = mu.predict_loss_projection(
            steps, ema_smooth, x_max, self.proj
        )
        stats          = mu.check_efficiency(popt, steps[-1], self.horizon)
        status_plateau = "\u2714 OUI" if stats["is_plateau"] else "\u2715 NON"
        info_text = (
            f"PREDICTION (Target: {self.horizon/1000:.0f}k)\n"
            f"---------------------------\n"
            f"Loss Horizon : {stats['loss_horizon']}\n"
            f"Slope/1k (now) : {stats['current_slope_per_1000']:.4f}\n"
            f"Slope/1k ({self.horizon/1000:.0f}k) : {stats['future_slope_per_1000']:.4f}\n"
            f"Plateau : {status_plateau}"
        )

        # ---- Projection LR et mix datasets ---------------------------------
        proj_x, proj_lr_y, proj_wiki, proj_cult, proj_litt = [], [], [], [], []
        s0 = steps[-1]
        for s in range(s0, x_max, 2):
            proj_x.append(s)
            proj_lr_y.append(mu.calcul_lr(s) * 1.e4)
            proj_wiki.append(
                (mu.calcul_ds(s - s0, (1 - cfg.MIXED - cfg.LITT), cfg.GRAD_ACCUM)
                 * cfg.BATCH_SIZE / cfg.TARGET_BLOCK_DS1 + monitor_ds1[-1]) % 1
            )
            proj_cult.append(
                (mu.calcul_ds(s - s0, cfg.MIXED, cfg.GRAD_ACCUM)
                 * cfg.BATCH_SIZE / cfg.TARGET_BLOCK_DS2 + monitor_ds2[-2]) % 1
            )
            proj_litt.append(
                (mu.calcul_ds(s - s0, cfg.LITT, cfg.GRAD_ACCUM)
                 * cfg.BATCH_SIZE / cfg.TARGET_BLOCK_DS3 + monitor_ds3[-1]) % 1
            )

        lines = []

        # ---- Tracé des courbes de loss ------------------------------------
        if smooth > 1:
            s_smooth  = steps[smooth - 1:]
            l_smooth  = mu.moving_average(val_loss, smooth)
            line_ax_1, = ax.plot(s_smooth, mu.moving_average(train_loss, smooth),
                                 label=f"Train (Smooth {smooth})", color="#3498db", lw=1.5)
            lines.append(ax.plot(s_smooth, l_smooth,
                                 label=f"Val (Smooth {smooth})", color="#e67e22", lw=1.5))
            ax.text(monitor_step_ds[-1], train_loss[-1],
                    s=f"\u25c4 {train_loss[-1]:.3f}", color="#3498db",
                    rotation=0, fontsize=9, horizontalalignment="left", va='center')
            ax.text(monitor_step_ds[-1], val_loss[-1],
                    s=f"\u25c4 {val_loss[-1]:.3f}", color="#e67e22",
                    rotation=0, fontsize=9, horizontalalignment="left", va='center')

            if val_loss_second is not None:
                s_sm_sec = steps_second[smooth - 1:]
                lines.append(ax.plot(s_sm_sec, mu.moving_average(val_loss_second, smooth),
                                     label="Model 2 (Comparison)", color="#fb59b6", lw=0.9, linestyle="--"))
                lines.append(ax.plot(steps_second, ema_smooth_2,
                                     label="EMA lissée second", color="#cf35b8", linestyle="-.", lw=1.))

            if val_loss_third is not None:
                s_sm_sec = steps_third[smooth - 1:]
                lines.append(ax.plot(s_sm_sec, mu.moving_average(val_loss_third, smooth),
                                     label="Model 1 (Comparison)", color="#5bf9b6", lw=0.9, linestyle="--"))
                lines.append(ax.plot(steps_third, ema_smooth_3,
                                     label="EMA lissée third", color="#70eef3", linestyle="-.", lw=1.))

            lines.append(ax.plot(steps, train_loss, color="#3498db", alpha=0.4, lw=1))
            lines.append(ax.plot(steps, val_loss,   color="#ff0020", alpha=0.4, lw=1))
            lines.append(ax.plot(steps, ema_smooth,
                                 label="EMA lissée",     color="#ff0020", linestyle="-.", lw=1.5))
            lines.append(ax.plot(steps_proj, ema_proj,
                                 label="EMA Projection", color="#ff0020", linestyle=":", lw=1.1))
            if raw:
                lines.append(ax.plot(monitor_step_ds[5:],
                                     mu.moving_average(monitor_loss, 5) * 1.,
                                     label="train: raw data", color="#000000", linestyle=":", lw=0.5))
        else:
            lines.append(ax.plot(steps, train_loss,
                                 label="Train Loss",       color="#3498db", lw=1.5))
            lines.append(ax.plot(steps, ema_train_loss,
                                 label="Train Loss - EMA", color="#3498db", linestyle="-.", lw=1.))

            if ds:
                lines.append(ax.plot(steps, train_loss_wiki,
                                     label="Train Loss Wikipédia",  color="Blue",   linestyle="--", lw=0.9, alpha=0.7))
                lines.append(ax.plot(steps, ema_train_loss_wiki,
                                     label="Wiki - Train EMA",      color="Blue",   linestyle="-.", lw=1.))
                lines.append(ax.plot(steps, val_loss_wiki,
                                     label="Val Loss Wikipédia",    color="Blue",   linestyle="--", lw=0.9, alpha=0.7))
                lines.append(ax.plot(steps, ema_val_loss_wiki,
                                     label="Wiki - EMA",            color="Blue",   linestyle="-.", lw=1.))
                lines.append(ax.plot(steps, train_loss_cult,
                                     label="Train Loss CulturaX",   color="Orange", linestyle="--", lw=0.9, alpha=0.7))
                lines.append(ax.plot(steps, ema_train_loss_cult,
                                     label="Cult - Train EMA",      color="Orange", linestyle="-.", lw=1.))
                lines.append(ax.plot(steps, val_loss_cult,
                                     label="Val Loss CulturaX",     color="Orange", linestyle="--", lw=0.9, alpha=0.7))
                lines.append(ax.plot(steps, ema_val_loss_cult,
                                     label="CulturaX - EMA",        color="Orange", linestyle="-.", lw=1.))
                lines.append(ax.plot(steps, train_loss_litt,
                                     label="Train Loss Littéraire", color="Green",  linestyle="--", lw=0.9, alpha=0.7))
                lines.append(ax.plot(steps, ema_train_loss_litt,
                                     label="Litt - Train EMA",      color="Green",  linestyle="-.", lw=1.))
                lines.append(ax.plot(steps, val_loss_litt,
                                     label="Val Loss Littéraire",   color="Green",  linestyle="--", lw=0.9, alpha=0.7))
                lines.append(ax.plot(steps, ema_val_loss_litt,
                                     label="Litteraire - EMA",      color="Green",  linestyle="-.", lw=1.))

            ax.text(monitor_step_ds[-1], train_loss[-1],
                    s=f"\u25c4 {train_loss[-1]:.3f}", color="#3498db",
                    rotation=0, fontsize=9, horizontalalignment="left", va='center')
            ax.text(monitor_step_ds[-1], val_loss[-1],
                    s=f"\u25c4 {val_loss[-1]:.3f}", color="#e67e22",
                    rotation=0, fontsize=9, horizontalalignment="left", va='center')
            lines.append(ax.plot(steps, val_loss,
                                 label="Val Loss",       color="#ff0020", lw=2))
            lines.append(ax.plot(steps, ema_smooth,
                                 label="Val Loss - EMA", color="#ff0020", linestyle="-.", lw=1.5))
            lines.append(ax.plot(steps_proj, ema_proj,
                                 label="Projection ...", color="#ff0020", linestyle=":", lw=1.1))
            if val_loss_second is not None:
                lines.append(ax.plot(steps_second, val_loss_second,
                                     label="Model v4", color="#fb59b6", linestyle="--", lw=0.9))

        if raw:
            lines.append(ax.plot(monitor_step_ds,    monitor_loss,
                                 label="Raw data",       color="#000000", linestyle=":", lw=0.5))
            lines.append(ax.plot(monitor_step_ds[5:], mu.moving_average(monitor_loss, 6) * 1,
                                 label="Raw data smooth", color="#000000", linestyle="solid", lw=0.5))
            lines.append(ax.plot(monitor_step_ds,    ema_smooth_raw,
                                 label="Raw data EMA",   color="#000000", linestyle="-.", lw=0.5))

        # ---- TARGET (ligne horizontale) ------------------------------------
        if (self.target is not None) and (self.target > y_min):
            ax.axhline(y=self.target, color="#aa55bb", alpha=0.7, linestyle="-.", lw=1.5)
            ax.text(max(500, x_min + 200), (self.target + 0.01),
                    s="Loss Cible à " + f"{self.target:.2f}",
                    color="#aa55bb", rotation=0, fontsize=9, horizontalalignment="left")

        # ---- Annotations événements ----------------------------------------
        if self.annot_event:
            for ev in self.annot_event:
                if ev["step"] >= x_min:
                    if (self.speed is not None) and (ev['step'] > steps[-1]):
                        elapse = (ev['step'] - steps[-1]) * self.speed
                        eta    = datetime.now() + timedelta(seconds=elapse)
                        ETA    = " (" + eta.strftime("%d %H:%M") + ")"
                    else:
                        ETA = ""
                    ax.axvline(x=ev["step"], color=ev.get("color", "black"),
                               linestyle="--", alpha=0.6, lw=ev.get("lw", 1))
                    ax.text(ev["step"], y_min + 0.06,
                            f"[{ev['step']:5d}] - " + ev["label"] + ETA,
                            rotation=90, color=ev.get("color", "black"), fontsize=9,
                            ha='right', verticalalignment="bottom")

        # ---- Compare point + lignes epoch ----------------------------------
        if compare:
            ax.axvline(x=compare, color="black", linestyle="--", alpha=0.7, lw=0.8)
            ax.text(compare + x_sens / 10, y_max, "Compare ",
                    rotation=90, color="black", fontsize=9, verticalalignment="top")
            for e in epoch_1:
                label = f"[{e}] Epoch - Dataset 1 (Wiki)"
                if e < x_max:
                    ax.axvline(x=e, color="blue", linestyle="--", alpha=0.7, lw=0.8)
                    ax.text(e, y_min + 0.06, label,
                            rotation=90, color="blue", fontsize=9,
                            ha='right', verticalalignment="bottom")
            for e in epoch_2:
                label = f"[{e}] Epoch - Dataset 2 (CulturaX)"
                if e < x_max:
                    ax.axvline(x=e, color="orange", linestyle="--", alpha=0.7, lw=0.8)
                    ax.text(e, y_min + 0.06, label,
                            rotation=90, color="orange", fontsize=9,
                            ha='right', verticalalignment="bottom")
            for e in epoch_3:
                label = f"[{e}] Epoch - Dataset 3 (Littéraire)"
                if e < x_max:
                    ax.axvline(x=e, color="green", linestyle="--", alpha=0.7, lw=0.8)
                    ax.text(e, y_min + 0.06, label,
                            rotation=90, color="green", fontsize=9,
                            ha='right', verticalalignment="bottom")

        # ---- Réglages finaux -----------------------------------------------
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_min, x_max)
        ax.set_title(
            f"{self.titre} | Step: {steps[-1]} | Gap: {gap[-1]:.4f}", fontsize=10
        )
        ax.grid(True, alpha=0.3)

        pu.add_label(ax, f"\u25c4 {monitor_loss[-1]:.3f}", monitor_loss[-1], offset=0, left=0.)
        pu.add_label(ax, f"\u25ba", train_loss[-1], offset=0, left=0., color="#3498db", textAlign="right")
        pu.add_label(ax, f"\u25ba", val_loss[-1],   offset=0, left=0., color="#e67e22", textAlign="right")
        pu.add_label(ax, f"\u25c4 {ema_smooth[-1]:.3f}", ema_smooth[-1], offset=0, left=0.,
                     color="#ff0000", textAlign="left")
        pu.add_label(ax, f"\u25ac", val_loss_wiki[-1], offset=0, left=0., color="Blue",   textAlign="center")
        pu.add_label(ax, f"\u25ac", val_loss_cult[-1], offset=0, left=0., color="Orange", textAlign="center")
        pu.add_label(ax, f"\u25ac", val_loss_litt[-1], offset=0, left=0., color="Green",  textAlign="center")

        # ---- Axe droit 1 : Dataset % ----------------------------------------
        ax_droite = ax.twinx()
        ax_droite.grid(False)
        y_1_max  = math.ceil(
            20 * min(1.0, 1.25 * max(max(monitor_ds1), max(monitor_ds2), max(monitor_ds3)))
        ) / 20.
        y_1_tick = y_1_max / 20.
        ax_droite.set_ylim(0.0, y_1_max)
        ax_droite.set_yticks(np.arange(0., y_1_max + epsilon, y_1_tick))
        ax_droite.tick_params(axis='y', colors='gray', labelsize=8, length=2, width=0.7)
        ax_droite.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f'{100*x:.1f}%')
        )
        lines.append(ax_droite.plot(monitor_step_ds, monitor_ds1,
                                    color="blue",   alpha=0.8, label="Wiki %", lw=0.8))
        ax_droite.text(monitor_step_ds[-1] + x_sens / 10, monitor_ds1[-1], "wikipédia",
                       color="blue",   fontsize=9, va='center', ha="left", zorder=100, alpha=0.9)
        lines.append(ax_droite.plot(monitor_step_ds, monitor_ds2,
                                    color="orange", alpha=0.7, label="Cult %", lw=0.8))
        ax_droite.text(monitor_step_ds[-1] + x_sens / 10, monitor_ds2[-1], "culturaX",
                       color="orange", fontsize=9, va='center', ha="left", zorder=100, alpha=0.9)
        lines.append(ax_droite.plot(monitor_step_ds, monitor_ds3,
                                    color="green",  alpha=0.7, label="Litt %", lw=0.8))
        ax_droite.text(monitor_step_ds[-1] + x_sens / 10, monitor_ds3[-1], "littéraire",
                       color="green",  fontsize=9, va='center', ha="left", zorder=100, alpha=0.9)

        pu.add_label(ax_droite, f"\u25c4 {monitor_ds1[-1]*100:.1f}%", monitor_ds1[-1], 0, color='blue')
        pu.add_label(ax_droite, f"\u25c4 {monitor_ds2[-1]*100:.1f}%", monitor_ds2[-1], 0, color='orange')
        pu.add_label(ax_droite, f"\u25c4 {monitor_ds3[-1]*100:.1f}%", monitor_ds3[-1], 0, color='green')

        # ---- Axe droit 2 : Gradient norm -------------------------------------
        ax_droite_2 = ax.twinx()
        ax_droite_2.grid(False)
        y_2_max = min(1.0, max(monitor_grad[-15:])*1.3)
        ax_droite_2.set_ylim(0., y_2_max)
        ax_droite_2.spines['right'].set_position(("outward", 34))
        ax_droite_2.set_yticks(np.arange(0., y_2_max + epsilon, 0.01))
        ax_droite_2.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
        ax_droite_2.tick_params(axis='y', which='major', colors='gray', labelsize=8, length=3.5, width=0.8)
        ax_droite_2.tick_params(axis='y', which='minor', colors='gray', length=2, width=0.5)
        lines.append(ax_droite_2.plot(monitor_step_ds, monitor_grad,
                                      color="#7B8c80", alpha=0.7, label="Gradient norm", lw=0.6))
        pu.add_label(ax_droite_2, f"\u25c4 {monitor_grad[-1]:.3f}", monitor_grad[-1], offset=34)

        # ---- Axe droit 3 : Learning Rate -------------------------------------
        ax_droite_3 = ax.twinx()
        ax_droite_3.grid(False)
        y_3_max = 3.5
        ax_droite_3.set_ylim(0.0, y_3_max)
        ax_droite_3.spines['right'].set_position(("outward", 65))
        ax_droite_3.set_yticks(np.arange(0., y_3_max + epsilon, 0.1))
        ax_droite_3.tick_params(axis='y', colors='gray', labelsize=8)
        ax_droite_3.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax_droite_3.tick_params(axis='y', which='major', colors='gray', labelsize=8, length=3.5, width=0.8)
        ax_droite_3.tick_params(axis='y', which='minor', colors='gray', length=2, width=0.5)
        lines.append(ax_droite_3.plot(monitor_step_ds, monitor_lr,
                                      color="#5b6c60", alpha=0.8, label="Learning Rate",
                                      linestyle='-', lw=.9))
        lines.append(ax_droite_3.plot(proj_x, proj_lr_y,
                                      color="#5b6c60", alpha=0.8, linestyle=':', lw=.9))
        lines.append(ax_droite.plot(proj_x, proj_wiki, color="blue",   alpha=0.8, linestyle=":", lw=0.5))
        lines.append(ax_droite.plot(proj_x, proj_cult, color="orange", alpha=0.8, linestyle=":", lw=0.5))
        lines.append(ax_droite.plot(proj_x, proj_litt, color="green",  alpha=0.8, linestyle=":", lw=0.5))
        pu.add_label(ax_droite_3, f"\u25c4 {monitor_lr[-1]:.2f}e-4", monitor_lr[-1], offset=65)

        # ---- Encart info (prediction) ----------------------------------------
        plt.text(0.4, 0.87, info_text, transform=plt.gca().transAxes,
                 fontsize=9, verticalalignment='bottom', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='#ccddff', edgecolor="#aabbdd", alpha=0.5))

        fig.t_line = ax_droite_3.annotate(
            "", xy=(0, 1), xycoords=('data'), xytext=(10, 10),
            textcoords='offset points', rotation=0, color="black",
            fontsize=7, verticalalignment="bottom",
            bbox=dict(facecolor='#ccddff', alpha=1., edgecolor='#aabbdd', boxstyle='square,pad=0.4'),
            ha='center', va='center', zorder=1000
        )

        # ---- Légende cliquable ----------------------------------------------
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_droite.get_legend_handles_labels()
        h3, l3 = ax_droite_2.get_legend_handles_labels()
        h4, l4 = ax_droite_3.get_legend_handles_labels()
        handles = h1 + h2 + h3 + h4
        labels  = l1 + l2 + l3 + l4

        leg = ax_droite_3.legend(handles, labels, loc="upper left", frameon=True)
        self.line_dict = {}
        for leg_line, leg_text, orig_line in zip(leg.get_lines(), leg.get_texts(), lines):
            leg_line.set_picker(True)
            leg_text.set_picker(True)
            self.line_dict[leg_text] = orig_line
            self.line_dict[leg_line] = orig_line

        map_line = {t.get_text(): (self.line_dict[t][0], t) for t in leg.get_texts()}
        mask = {
            'Val Loss Wikipédia', 'Val Loss CulturaX', 'Val Loss Littéraire',
            'Train Loss CulturaX', 'Train Loss Wikipédia', 'Train Loss Littéraire',
            'Wiki - Train EMA', 'Cult - Train EMA', 'Litt - Train EMA',
            'Model v4', 'Wiki %', 'Cult %', 'Litt %',
            'Train Loss', 'Val Loss', 'Raw data', 'Raw data smooth'
        }
        for l in mask:
            line, legend = map_line[l]
            pu.set_off(line, legend)

        fig.canvas.mpl_connect('pick_event', self.on_pick)

        pu.copyright(ax, y_min)
        plt.savefig("model/screenshot/analyse_poussin_results.png")
        plt.show()

        # ---- Stats dans le terminal -----------------------------------------
        print("-" * 30)
        print(f"Dernier Step: {steps[-1]} | Gap actuel: {gap[-1]:.4f} | EMA loss : {ema_last:.3f}")
        if val_loss_second is not None:
            print(f"COMPARE POINT ANALYSIS: {steps[end]} / {steps_second[end_second]}")
            print(f"  - Val Loss: {val_loss[-1]:.4f} vs {val_loss_second[end_second]:.4f}")
            print(f"  - EMA Loss: {ema_loss:.3f} vs {ema_loss_second:.3f}")
        print("-" * 100)

        tokens_m5 = dl.convert_steps_list(steps, "model_5")


# ===========================================================================
# BACKWARD COMPAT  —  wrapper fonctionnel conservé
# ===========================================================================

def plot_poussin_gap(
    y_min=2.6, y_max=3.3, x_max=70000, x_min=0, smooth=5,
    log_path="model/my_wiky_history.json",
    second="save/model_3/my_wiky_history.json",
    third=None, compare=None, annot_event=None, target=None,
    speed=None, raw=False, titre="Training GPT Mac-M1",
    horizon=30000, proj=10, phase_train=None, ds=False
):
    """Wrapper conservé pour compatibilité ascendante avec l'ancien appel fonctionnel."""
    PoussinPlotter(
        y_min=y_min, y_max=y_max, x_max=x_max, x_min=x_min, smooth=smooth,
        log_path=log_path, second=second, third=third, compare=compare,
        annot_event=annot_event, target=target, speed=speed, raw=raw,
        titre=titre, horizon=horizon, proj=proj, phase_train=phase_train, ds=ds
    ).plot()
