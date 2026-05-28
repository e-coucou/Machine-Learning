
    def _setup_data(self,data_dir):
        # --- CHARGEMENT DES DONNÉES BINAIRES (Nouveau) ---
        # On utilise memmap pour lire le fichier sur le disque sans charger la RAM
        train_path = os.path.join(data_dir, 'train_wiki.bin')
        val_path = os.path.join(data_dir, 'val_wiki.bin')
        train_cult_path = os.path.join(data_dir, 'train_culturax.bin')
        val_cult_path = os.path.join(data_dir, 'val_culturax.bin')
        train_litt_path = os.path.join(data_dir, 'train_litteraire.bin')
        val_litt_path = os.path.join(data_dir, 'val_litteraire.bin')
        
        if os.path.exists(train_path):
            self.train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
            print(f"🚀 Dataset Train: {len(self.train_data)/1e6:.2f}M tokens.")
            self.val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
            print(f"🚀 Dataset Val: {len(self.val_data)/1e6:.2f}M tokens.")
        else:
            print(f"⚠️ Fichiers binaires introuvables dans {data_dir}")      

        if self.cult_data and os.path.exists(train_cult_path):
            self.train_data_cult = np.memmap(train_cult_path, dtype=np.uint16, mode='r')
            print(f"🚀 Dataset CulturaX: {len(self.train_data_cult)/1e6:.2f}M tokens.")
            self.val_cult = np.memmap(val_cult_path, dtype=np.uint16, mode='r')
            print(f"🚀 Dataset Val: {len(self.val_cult)/1e6:.2f}M tokens.")
        else:
            self.train_data_cult = None
            print("ℹ️ Mode Source Unique : Wiki uniquement.")    

        if self.litt_data and os.path.exists(train_litt_path):
            self.train_data_litt = np.memmap(train_litt_path, dtype=np.uint16, mode='r')
            print(f"🚀 Dataset Litteraire: {len(self.train_data_litt)/1e6:.2f}M tokens.")
            self.val_litt = np.memmap(val_litt_path, dtype=np.uint16, mode='r')
            print(f"🚀 Dataset Val: {len(self.val_litt)/1e6:.2f}M tokens.")
        else:
            self.train_data_litt = None
            print(f"🚀 Dataset Train: {len(self.train_data)/1e6:.2f}M tokens.")


    def get_batch_from_source(self, data):
        """
        Version optimisée de la fonction pour piocher dans n'importe quel memmap.
        """
        # 1. Générer tous les indices d'un coup
        ix = np.random.randint(0, len(data) - self.config['block_size'], (self.params['batch_size'],))
        
        # 2. Grille d'indices (la "magie" NumPy)
        offsets = np.arange(self.config['block_size'])
        indices = ix[:, None] + offsets 
        
        x_np = data[indices]
        y_np = data[indices + 1]

        # Conversion en Tenseur (on reste en long pour la CrossEntropy)
        # Note: On passe en .long() car la CrossEntropy ne prend pas le uint16
        x = torch.from_numpy(x_np).to(self.device).long()
        y = torch.from_numpy(y_np).to(self.device).long()

        del x_np, y_np # zut oublié de détruire ces derniers !!

        return x, y


    def get_batch_from_source(self, data):
        """
        Version optimisée de la fonction pour piocher dans n'importe quel memmap.
        Correction apporté pour éviter les fuites mémoire entre Numpy et Pytorch
        """
        # 1. Générer tous les indices d'un coup
        ix = np.random.randint(0, len(data) - self.config['block_size'], (self.params['batch_size'],))
        
        # 2. Grille d'indices (la "magie" NumPy)
        offsets = np.arange(self.config['block_size'])
        indices = ix[:, None] + offsets 
        # np.array fait une copy, on coupe la liaison avec le disque SSD
        x_np = np.array(data[indices], dtype=np.int64)
        y_np = np.array(data[indices + 1], dtype=np.int64)

        # Conversion en Tenseur (on reste en long pour la CrossEntropy)
        # Note: On passe en .long() car la CrossEntropy ne prend pas le uint16
#        x = torch.from_numpy(x_np).to(self.device).long()
#        y = torch.from_numpy(y_np).to(self.device).long()
#        x = torch.tensor(x_np, device = self.device)
#        y = torch.tensor(y_np, device = self.device)
        x = torch.as_tensor(x_np, device = self.device)
        y = torch.as_tensor(y_np, device = self.device)

#        del x_np, y_np # zut oublié de détruire ces derniers !!

        return x, y



    def estimate_loss_bin_v6(self, eval_iters=30):
        """
        Fonction helper pour estimer le loss sans dropout (mode eval).
        eval_iters: nombre de batchs pour moyenner et avoir un score stable.
        ici en V6 on va calculer par rapport aux 3 datasets
        """
        out = {}
        self.model.eval() # Désactive Dropout
        # on fixe les datasets
        # Dictionnaire des sources de validation à tester
        val_sources = {
            'wiki': self.val_data,
            'cult': self.val_cult,
            'litt': self.val_litt
            }
        train_sources = {
            'wiki': self.train_data,
            'cult': self.train_data_cult,
            'litt': self.train_data_litt
            }

        # 1. Purge préventive avant l'effort d'évaluation
        gc.collect()           # Libère la RAM CPU (NumPy/Tensors CPU)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # 2. Utilisation de inference_mode (plus rapide que no_grad sur Metal)
        with torch.inference_mode(): # ce code crash !!
            with torch.autocast(device_type='mps', dtype=torch.float16):
#            with torch.no_grad():
            #2.1/ le train pqr source new version
                for name, data_source in train_sources.items():
                    if data_source is not None:
                        t_losses = torch.zeros(eval_iters)
                        for k in range(eval_iters):
                            X, Y = self.get_batch_from_source(data_source)
                            _, loss = self.model(X, Y)
                            t_losses[k] = loss.item()
                        out[f"t_{name}"] = t_losses.mean().item()

            # 2.2 /Validation détaillée par domaine
                for name, data_source in val_sources.items():
                    if data_source is not None:
                        v_losses = torch.zeros(eval_iters)
                        for k in range(eval_iters):
                            # On utilise ta logique "magie numpy" directement sur le source
                            X, Y = self.get_batch_from_source(data_source)
                            _, loss = self.model(X, Y)
                            v_losses[k] = loss.item()
                        out[f'v_{name}'] = v_losses.mean().item()

        # Calcul de la Loss globale pondérée pour le graphe principal
            out['train'] = (self.wiki_ratio * out.get('t_wiki', 0) +
                            self.cult_ratio * out.get('t_cult', 0) + 
                            self.litt_ratio * out.get('t_litt', 0))

            out['val']   = (self.wiki_ratio * out.get('v_wiki', 0) + 
                            self.cult_ratio * out.get('v_cult', 0) + 
                            self.litt_ratio * out.get('v_litt', 0))
       
        self.model.train() # Réactive Dropout
            
        # 3. Purge préventive avant l'effort d'évaluation
        gc.collect()           # Libère la RAM CPU (NumPy/Tensors CPU)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        status, pression, mem_rss = self.log_memory_status()

        return out, status, pression, mem_rss
