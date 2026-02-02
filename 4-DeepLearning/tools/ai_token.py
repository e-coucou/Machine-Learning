from collections import Counter
from itertools import chain
import json, base64, time, re, random, colorsys
import regex as rex
from functools import lru_cache

import numpy as np
from IPython.display import HTML, display

class BPETokenizer:
    """
    Tokenizer BPE (Byte Pair Encoding)
    Encode/décode du texte en utilisant l'algorithme BPE
    """
    def __init__(self, fileName=None, texte=None, addToken=1000):
        self.addToken = addToken
        self.fileName = fileName
        self.merges = {}
        self.vocab = {}
        self.stats = {
            'len_raw': 0,
            'len_init': 0,
            'len_final': 0,
            'compression_ratio': 0,
            'unique_tokens': 0,
            'len_cleaned': 0,
            'temps_encodage':0,
            'temps_train':0,
            'temps_decodage':0,
            'Fast_encode':0,
            'temps_train_opt':0
        }
        self.text_raw = None
        if fileName is not None:
            self.read_file()
        if texte is not None:
            self.text_raw = texte
        if self.text_raw is not None:
            self.tokenize(option=5)
        # else: # inutile ou alors créer un mode verbose
        #     print("vous devrez soit lire un fichier soit envoyer un texte pour lancer le tokenizer")
        # self.preprocess_text()
        # # Convertir le texte en liste de tokens (bytes)
        # #self.ids = [list(text.encode('utf-8')) for text in self.text_cleaned]
        # self.tokenize(option=5) # en francais
        # self.token = [ list(map(int,i.encode('utf-8'))) for i in self.token_char ]
        # # self.ids = [ list(map(int,i.encode('utf-8'))) for i in self.token_char ]

    def read_file(self):
        with open(self.fileName,'r',encoding='utf-8') as f:
            self.text_raw = f.read()
        self.stats['len_raw'] = len(self.text_raw)

    def preprocess_text(self):
        # 1. Remplacements de caractères (très rapide)
        replacements = {
            '—': '-', '«': '"', '»': '"', '“': '"', '”': '"',
            '[': '(', ']': ')', '{': '(', '}': ')',
            '…': '...', '’': "'", '‘': "'"
        }
        for old, new in replacements.items():
            self.text_raw = self.text_raw.replace(old, new)

        # 2. Filtrage global via Regex (beaucoup plus rapide que la boucle for)
        # On définit ce qu'on veut GARDER
        keep_pattern = r'[^a-zA-Z0-9 .,;:!?\'\"\nàâçèéêëîïôùûÀÂÇÉÈÊËÎÏÔÙÛœŒ()-]'
        self.text_cleaned = re.sub(keep_pattern, '', self.text_raw)

        # 3. Normalisation de la ponctuation (pas d'espace avant)
        self.text_cleaned = re.sub(r'\s+([.,;:!?])', r'\1', self.text_cleaned)

        # 4. Normalisation finale des espaces
        # On remplace les tabs et espaces multiples par un seul espace
        self.text_cleaned = re.sub(r'[ \t]+', ' ', self.text_cleaned)
        # On limite à maximum 2 sauts de ligne (garde les paragraphes, vire le vide)
        self.text_cleaned = re.sub(r'\n{3,}', '\n\n', self.text_cleaned)
        
        self.text_cleaned = self.text_cleaned.strip()
        
        self.stats['len_cleaned'] = len(self.text_cleaned)
        return self.text_cleaned

    def preprocess_text_lent(self): # Nettoie un fichier texte pour l'entraînement NLP
        # Nettoyage
        self.text_cleaned = re.sub(r'[ \t]+', ' ', self.text_raw)              # Espaces multiples
        self.text_cleaned = re.sub(r'\n{3,}', '\n\n', self.text_cleaned)           # Lignes vides
        self.text_cleaned = re.sub(r'(?m)^[ \t]+|[ \t]+$', '', self.text_cleaned)  # Espaces début/fin ligne
        self.text_cleaned = re.sub(r'\s+([.,;:!?])', r'\1', self.text_cleaned)     # Espace avant ponctuation
        self.text_cleaned = self.text_cleaned.strip()
        # 1. Remplacements intelligents
        self.text_cleaned = self.text_cleaned.replace('—', '-')
        self.text_cleaned = self.text_cleaned.replace('«', '"').replace('»', '"')
        self.text_cleaned = self.text_cleaned.replace('[', '(').replace(']', ')')
        self.text_cleaned = self.text_cleaned.replace('{', '(').replace('}', ')')
        self.text_cleaned = self.text_cleaned.replace(''', "'").replace(''', "'")
        self.text_cleaned = self.text_cleaned.replace('…', '...')
        self.text_cleaned = self.text_cleaned.replace('«', '"').replace('»', '"')
    
        # 2. Filtrer caractères
        allowed = set(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789 .,;:!?\'\"-\n'
            'àâçèéêëîïôùûÀÂÇÉÈÊËÎÏÔÙÛœŒ'
            '()'
        )
        #self.text_cleaned = ''.join(c for c in self.text_cleaned if c in allowed)
        # Tout ce qui n'est pas dans mon set "allowed" est supprimé
        # (Note: il faut construire le regex avec précaution)
        pattern = re.compile(r'[^a-zA-Z0-9 .,;:!?\'\"-\nàâçèéêëîïôùûÀÂÇÉÈÊËÎÏÔÙÛœŒ()]')
        self.text_cleaned = pattern.sub('', self.text_cleaned)
        
        # 3. Normaliser espaces
        self.text_cleaned = re.sub(r' +', ' ', self.text_cleaned)
        self.text_cleaned = re.sub(r'\n{3,}', '\n\n', self.text_cleaned)
        self.text_cleaned = re.sub(r'(?m)^[ ]+|[ ]+$', '', self.text_cleaned)
                
        self.text_cleaned = self.text_cleaned.strip()    
        # Stats après nettoyage
        self.stats['len_cleaned'] = len(self.text_cleaned)
        print(f"Nettoyage terminé:")
        return self.text_cleaned
        
    def _get_stats(self): # Compte les paires de tokens les plus fréquentes
        pairs = Counter()
        for row in self.ids:
            for pair in zip(row, row[1:]):
                pairs[pair] += 1
        return pairs

    def _get_stats_data(self, data):
            if len(data) < 2: return None
            # On crée les paires via décalage (shift)
            first, second = data[:-1], data[1:]
            # On ignore les séparateurs -1
            mask = (first != -1) & (second != -1)
            if not np.any(mask): return None
            
            # On combine en int64 pour la vitesse
            combined = (first[mask].astype(np.int64) << 32) | second[mask].astype(np.int64)
            unique, counts = np.unique(combined, return_counts=True)
            
            best_val = unique[np.argmax(counts)]
            return (int(best_val >> 32), int(best_val & 0xFFFFFFFF))

    def _get_max_stats(self): # Retourne la paire la plus fréquente
        pairs = self._get_stats()
        return pairs.most_common(1)[0][0] if pairs else None
    
    def _get_max_stats_numpy(self):
        # 1. On prépare une liste plate avec un séparateur (-1)
        # Note: On fait ça une fois ou on maintient le tableau numpy à jour
        flat_ids = []
        for row in self.ids:
            flat_ids.extend(row)
            flat_ids.append(-1)
        
        data = np.array(flat_ids, dtype=np.int32)
        
        # 2. On crée les paires (i, i+1)
        # On regarde les éléments de 0 à n-1 et de 1 à n
        first_el = data[:-1]
        second_el = data[1:]
        
        # 3. On masque les paires qui contiennent le séparateur -1
        mask = (first_el != -1) & (second_el != -1)
        
        # 4. ASTUCE : On combine deux int32 en un seul int64 pour un comptage ultra-rapide
        # Cela évite de gérer des tuples complexes
        combined = (first_el[mask].astype(np.int64) << 32) | second_el[mask].astype(np.int64)
        
        if len(combined) == 0:
            return None

        # 5. On compte les occurrences
        unique, counts = np.unique(combined, return_counts=True)
        
        # 6. On récupère la paire la plus fréquente
        max_idx = np.argmax(counts)
        best_combined = unique[max_idx]
        
        # On décode le int64 pour retrouver le tuple (id1, id2)
        pair = (int(best_combined >> 32), int(best_combined & 0xFFFFFFFF))
        
        return pair
    
    def tokenize(self,option=5):
        self.preprocess_text()

        pattern  = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pattern2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pattern3 = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
        GPT4_SP  = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        GPT_EP  = r"""(?i:[lcdtmnsj]|qu)'|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        if (option==1):
            patG = rex.compile(pattern)
        elif (option==2):
            patG = rex.compile(pattern2)
        elif (option==3):
            patG = rex.compile(pattern3)
        elif (option==4):
            patG = rex.compile(GPT4_SP)
        else:
            patG = rex.compile(GPT_EP)
        self.token_char = rex.findall(patG, self.text_cleaned)
        self.token = [ list(map(int,i.encode('utf-8'))) for i in self.token_char ]
        return self.token_char

    def _merge_pair(self, pair, idx): #Fusionne une paire de tokens dans l'array
        merged = []
        pair_tuple = pair
        for row in self.ids:
            merged_row = []
            i = 0
            row_len = len(row)
            while i < row_len:
                if i < row_len - 1 and row[i] == pair_tuple[0] and row[i+1] == pair_tuple[1]:
                    merged_row.append(idx)
                    i += 2
                else:
                    merged_row.append(row[i])
                    i += 1
            merged.append(merged_row)
        return merged
    
    def train_old(self, verbose=True): #Entraîne le tokenizer BPE
        start_time = time.time()
        self.stats['len_init'] = sum(len(row) for row in self.token)
        if verbose:
            print(f"Démarrage du BPE avec {self.addToken} merges...")   
        for i in range(self.addToken):
            pair = self._get_max_stats_numpy()
#            pair = self._get_max_stats()
            if not pair:
                if verbose:
                    print(f"Arrêt anticipé à {i} merges (pas de paires trouvées)")
                break
            
            idx = 256 + i
            self.ids = self._merge_pair(pair, idx)
            self.merges[pair] = idx
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i + 1}/{self.addToken} merges effectués...")
        elapse_time = time.time()-start_time
        # Calculs finaux
        self.stats['len_final'] = sum(len(row) for row in self.ids)
        self.stats['unique_tokens'] = len(set(chain(*self.ids)))
        self.stats['compression_ratio'] = self.stats['len_init'] / self.stats['len_final']
        self.stats['temps_train'] = elapse_time
        self._vocab()
        if verbose:
            self._print_stats()

    def train_optimiser(self, verbose=False):
        self.stats['len_init'] = sum(len(row) for row in self.token)
        start_time = time.time()
        # 1. Préparation : On aplatit tout en un seul vecteur int32
        # On insère un séparateur (ex: -1) entre chaque ligne pour éviter les fusions inter-lignes
        flat_list = []
        for row in self.token:
            flat_list.extend(row)
            flat_list.append(-1)
        
        data = np.array(flat_list, dtype=np.int32)
        
        if verbose:
            print(f"Entraînement sur {len(data)} tokens...")

        for i in range(self.addToken):
            # --- ÉTAPE A : COMPTAGE VECTORISÉ ---
            # On décale le tableau pour créer des paires
            first_el = data[:-1]
            second_el = data[1:]
            
            # On ignore les paires contenant le séparateur
            valid_mask = (first_el != -1) & (second_el != -1)
            
            if not np.any(valid_mask):
                break

            # Bit-shifting pour transformer (int32, int32) en un unique int64
            combined = (first_el[valid_mask].astype(np.int64) << 32) | second_el[valid_mask].astype(np.int64)
            
            # Comptage ultra-rapide avec NumPy
            unique, counts = np.unique(combined, return_counts=True)
            best_idx = np.argmax(counts)
            best_combined = unique[best_idx]
            
            # Extraire la paire gagnante
            pair = (int(best_combined >> 32), int(best_combined & 0xFFFFFFFF))
            new_token = 256 + i
            self.merges[pair] = new_token
            
            # --- ÉTAPE B : FUSION VECTORISÉE ---
            # On cherche où se trouve la paire dans le tableau original
            # Attention : on doit recalculer le masque sur la taille totale de 'data'
            match_mask = (data[:-1] == pair[0]) & (data[1:] == pair[1])
            
            # Gestion des chevauchements (ex: 'aaa' -> 'aa' + 'a', pas deux fusions)
            # On désactive le match suivant si on vient d'en trouver un
            match_indices = np.where(match_mask)[0]
            if len(match_indices) > 0:
                # Filtrer les indices successifs pour éviter les doubles fusions
                keep = np.ones(len(match_indices), dtype=bool)
                for j in range(len(match_indices) - 1):
                    if match_indices[j+1] == match_indices[j] + 1:
                        keep[j+1] = False
                match_indices = match_indices[keep]

            if len(match_indices) == 0:
                break

            # Reconstruction du tableau : on retire le deuxième élément de chaque paire fusionnée
            # et on remplace le premier par le nouveau token
            data[match_indices] = new_token
            
            # Masque pour supprimer les éléments fusionnés (les seconds éléments de la paire)
            to_remove = match_indices + 1
            data = np.delete(data, to_remove)

            if verbose and (i + 1) % 10 == 0:
                print(f"Merge {i+1}/{self.addToken} : {pair} -> {new_token} (Taille: {len(data)})")

        self.stats['temps_train_opt'] = time.time() - start_time
        return data

    def train(self, verbose=False): # train optimizer version 2
        """
        Version optimisée pour Mac M1 (Architecture Unified Memory).
        Utilise NumPy pour le comptage vectorisé et le remplacement par masque.
        """
        self.stats['len_init'] = sum(len(row) for row in self.token)
        start_time = time.time()
        # 1. RESET COMPLET : On repart sur une base saine
        self.merges = {}
        self.vocab = {}
        
        # 2. PRÉPARATION DES DONNÉES
        # On transforme les listes de listes en un seul tableau NumPy contigu
        # On insère -1 comme séparateur de ligne
        flat_ids = []
        for row in self.token:
            flat_ids.extend(row)
            flat_ids.append(-1)
        
        # On utilise int32 pour supporter des IDs > 255
        data = np.array(flat_ids, dtype=np.int32)
        
        # 3. INITIALISATION DU COMPTEUR D'ID
        # C'est ici qu'on garantit que 434 vient après 433 (et pas 4013)
        next_token_id = 256
        
        # 4. BOUCLE PRINCIPALE D'ENTRAÎNEMENT
        for i in range(self.addToken):
            if len(data) < 2:
                break
                
            # --- A. COMPTAGE RAPIDE (Bit-shifting) ---
            first = data[:-1]
            second = data[1:]
            # On ignore les paires à cheval sur un séparateur -1
            mask = (first != -1) & (second != -1)
            
            if not np.any(mask):
                break
                
            # On fusionne deux int32 en un seul int64 pour np.unique
            combined = (first[mask].astype(np.uint64) << 32) | (second[mask].astype(np.uint64) & 0xFFFFFFFF)
#            combined = (first[mask].astype(np.int64) << 32) | second[mask].astype(np.int64)
            unique, counts = np.unique(combined, return_counts=True)
            
            # --- B. SÉLECTION DE LA PAIRE ---
            max_idx = np.argmax(counts)
            best_combined = unique[max_idx]
            best_count = counts[max_idx]
            
            # On décode la paire
            val = int(best_combined)
            pair = ((val >> 32), (val & 0xFFFFFFFF))
            
            # --- C. FUSION DANS LE TABLEAU (Le "Merge") ---
            # On cherche où se trouve cette paire précise
            match_mask = (data[:-1] == pair[0]) & (data[1:] == pair[1])
            indices = np.where(match_mask)[0]
            
            if len(indices) > 0:
                # --- NOUVELLE LOGIQUE DE SÉCURITÉ (Sûre à 100%) ---
                if len(indices) > 1:
                    keep = []
                    last_idx = -2 # Pour être sûr de prendre le premier
                    for idx in indices:
                        # Si l'indice actuel est collé au précédent qu'on a gardé,
                        # on l'ignore car le caractère a déjà été "consommé" par la fusion
                        if idx >= last_idx + 2:
                            keep.append(idx)
                            last_idx = idx
                    indices = np.array(keep)
                
                self.merges[pair] = next_token_id
                # Remplacement vectorisé
                data[indices] = next_token_id
                # On supprime les seconds éléments des paires fusionnées
                data = np.delete(data, indices + 1)
                
                if verbose and (next_token_id % 100 == 0):
                    print(f"Token {next_token_id} créé (fréquence de la paire : {best_count})")
                
                # Incrémentation propre
                next_token_id += 1
            else:
                break

        # 5. FINALISATION
        # On sauvegarde le tableau fusionné et on génère le vocabulaire
        self.stats['temps_train_opt'] = time.time() - start_time
        self.ids = (data[data != -1]).tolist()
        self.enc = (data[data != -1]).tolist()
        self.stats['len_final'] = len(self.ids)
        self.stats['unique_tokens'] = len(set(self.ids))
        self.stats['compression_ratio'] = self.stats['len_init'] / self.stats['len_final']
        self._vocab()
        
        if verbose:
            print(f"Entraînement terminé. Dernier token ID : {next_token_id - 1}")
        
        return self.ids

    def _print_stats(self): # Affiche les statistiques
        print("\n" + "="*60)
        print(f"Statistiques Nettoyage")
        print("."*30)
        print(f"  Caractères: {self.stats['len_raw']:,} → {self.stats['len_cleaned']:,} ({self.stats['len_cleaned']/self.stats['len_raw']*100:.1f}%)")
        print("\n" + "="*60)
        print("Statistiques BPE")
        print("."*30)
        print(f"Tokens avant BPE:        {self.stats['len_init']:,}")
        print(f"Tokens après BPE:        {self.stats['len_final']:,}")
        print(f"Compression:             {self.stats['compression_ratio']:.2f}x")
        print(f"Tokens uniques:          {self.stats['unique_tokens']}")
        print(f"Merges effectués:        {len(self.merges)}")
        print("\n" + "="*60)
        print("Statistiques Temps")
        print("."*30)
        print(f"Encodage effectué:       {self.stats['temps_encodage']:.3f}s")
        print(f"Fast Encodage effectué:  {self.stats['Fast_encode']:.3f}s")
        print(f"Décodage effectué:       {self.stats['temps_decodage']:.3f}s")
        print(f"Train effectué:          {self.stats['temps_train']:.3f}s")
        print(f"Train Optimisé effectué: {self.stats['temps_train_opt']:.3f}s")
        print("="*60 + "\n")

    def _vocab(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self._compute_pattern()
    
    def encode_old(self, input= None, vocab = None): # Encode un texte en tokens BPE
        if (vocab==None):
            vocab_ = self.vocab
        else:
            vocab_ = vocab
        if (input==None):
            input_ = self.token
        else:
            input_ = input
        self.stats['len_init'] = sum(len(row) for row in self.token)
        start_time = time.time()
        # Créer un dictionnaire inverse {bytes_sequence: token_id}
        inverse_vocab = {v: k for k, v in vocab_.items()}
    
        # Trier par longueur décroissante pour matcher les séquences les plus longues d'abord
        sorted_pairs = sorted(inverse_vocab.items(), key=lambda x: len(x[0]), reverse=True)
        
        encoded_array = []
    
        # Traiter chaque ligne du tableau
        for row in input_:
            # Convertir la liste d'entiers en bytes
            row_bytes = bytes(row)
            encoded_row = []
            i = 0
            # Parcourir les bytes et chercher les tokens correspondants
            while i < len(row_bytes):
                matched = False
                # Essayer les tokens les plus longs d'abord
                for token_bytes, token_id in sorted_pairs:
                    # Vérifier si les bytes actuels correspondent au token
                    if row_bytes[i:i+len(token_bytes)] == token_bytes:
                        encoded_row.append(token_id)
                        i += len(token_bytes)
                        matched = True
                        break
                
                # Si pas de correspondance, prendre 1 byte à la fois
                if not matched:
                    encoded_row.append(row_bytes[i:i+1][0])
                    i += 1
            
            encoded_array.append(encoded_row)
        elapse_time = time.time() - start_time
        print("Encodage en ", elapse_time," ms")
        self.stats['len_final'] = sum(len(row) for row in self.ids)
        self.stats['unique_tokens'] = len(set(chain(*self.ids)))
        self.stats['compression_ratio'] = self.stats['len_init'] / self.stats['len_final']
        self.stats['temps_encodage'] = elapse_time
        self.ids = encoded_array # mise à jour
        return encoded_array
    
    def get_array(self, input):
        return list(chain(*input))
    
    def decode(self, array_=None, vocab=None): # Décode une liste de tokens en texte
        if (vocab==None):
            vocab_ = self.vocab
        else:
            vocab_ = vocab
        if (array_==None):
            array = self.get_array(self.ids)
        else:
            array = array_
        start_time = time.time()
        decoded_bytes = bytearray()
        for idx in array:
            decoded_bytes.extend(vocab_[idx])
        elapse_time = time.time() - start_time
        self.stats['temps_decodage'] = elapse_time
        return decoded_bytes.decode('utf-8', errors='replace')

    def save_vocab(self, filepath='tokenizer.json'):
        vocab_dict = {}    
        for idx, byte_val in self.vocab.items():
            vocab_dict[str(idx)] = base64.b64encode(byte_val).decode('ascii')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
        print(f"Vocabulaire sauvegardé dans {filepath}")
            
    def load_vocab(self, filepath='tokenizer.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        self.vocab = {}
        for idx_str, string_val in vocab_dict.items():
            self.vocab[int(idx_str)] = base64.b64decode(string_val)
        self._compute_pattern()
        print(f"Vocabulaire chargé depuis {filepath}")

    def save_merges(self, filename='data/tokenizer.json'):
        # On sauve les merges, car ils permettent de reconstruire le vocab
        m = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(filename, 'w') as f:
            json.dump(m, f)

    def load_merges(self, filename='data/tokenizer.json'):
        with open(filename, 'r') as f:
            m = json.load(f)
        self.merges = {tuple(map(int, k.split(','))): v for k, v in m.items()}
        self._vocab()

    def display_mot(self, array, vocab = None):
        html = '<div style="font-family: monospace; font-size: 14px; line-height: 1.8;">'
        if (vocab==None):
            vocab_ = self.vocab
        else:
            vocab_ = vocab
        t = [ self.decode(r, vocab=vocab_) for r in array ]
        txt = ''.join(t)
        for text in t:
            # Générer une couleur aléatoire
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            html += f'<span style="color: #fff; background-color: {color}; padding: 2px 4px; margin: 2px; border-radius: 3px;">{text}</span>'
        html += '</div>'
        display(HTML(html))
        print(f"\n✓ {len(t)} tokens affichés avec couleurs aléatoires")   

    def display_token(self, array_=None, vocab = None):
        # Sécurité pour transformer une liste en array si nécessaire
        array = np.array(array_) if array_ is not None else self.enc

        html = '<div style="font-family: monospace; font-size: 14px; line-height: 1.8;">'
        if (vocab==None):
            vocab_ = self.vocab
        else:
            vocab_ = vocab
        t = [ self.decode([r], vocab=vocab_) for r in array[:500] if (r != -1)] # au cas le dernier -1 ne serait pas masqué
        txt = ''.join(t)

        for text in t:
            # Générer une couleur aléatoire
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            html += f'<span style="color: #fff; background-color: {color}; padding: 2px 4px; margin: 2px; border-radius: 3px;">{text}</span>'
            
        html += '</div>'
        display(HTML(html))
        print(f"\n✓ {len(t)} tokens affichés avec couleurs aléatoires")   

    def _compute_pattern(self):
        # On inverse le vocabulaire pour avoir {bytes: id}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # On pré-calcule le pattern Regex
        # 1. On trie les séquences d'octets par longueur décroissante
        # 2. On les "échappe" pour éviter les caractères spéciaux regex
        # 3. On finit par . pour attraper n'importe quel octet restant
        sorted_bytes = sorted(self.inverse_vocab.keys(), key=len, reverse=True)
        
        # Construction du pattern : (token1|token2|token3|.)
        # Utilisation de re.escape pour les bytes et jointure par le pipe OR
        pattern_bytes = b'|'.join(re.escape(b) for b in sorted_bytes) + b'|.'
        self.compiled_pattern = re.compile(pattern_bytes)    

    def fast_encode(self, input_):
        start_time = time.time()
        encoded_array = []
        # Accès local pour plus de vitesse dans la boucle
        pattern = self.compiled_pattern
        inv_vocab = self.inverse_vocab
        
        for row in input_:
            row_bytes = bytes(row)
            # findall trouve tous les tokens correspondants en un seul passage (C-level)
            tokens = pattern.findall(row_bytes)
            
            # Conversion des morceaux de bytes en IDs
            # Si le token est dans inv_vocab, on prend son ID, sinon c'est un octet seul
            encoded_row = [inv_vocab.get(t, t[0]) for t in tokens]
            encoded_array.append(encoded_row)
        
        elapse_time = time.time() - start_time
        self.stats['Fast_encode'] = elapse_time
        return encoded_array        

    def encode(self, text): # encodeur with numpy
#        self.stats['len_init'] = sum(len(row) for row in self.token)
        start_time = time.time()
        # 1. Convertir le texte en tableau d'octets (0-255)
        if isinstance(text, str):
            data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.int32)
        else:
            data = np.array(list(text), dtype=np.int32)
        # 2. Appliquer chaque fusion apprise dans l'ordre
        for (p0, p1), new_id in self.merges.items():
            if len(data) < 2:
                break
                
            # Trouver où la paire (p0, p1) apparaît
            mask = (data[:-1] == p0) & (data[1:] == p1)
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                continue

            # --- NOUVELLE LOGIQUE DE SÉCURITÉ (Sûre à 100%) ---
            if len(indices) > 1:
                keep = []
                last_idx = -2 # Pour être sûr de prendre le premier
                for idx in indices:
                    # Si l'indice actuel est collé au précédent qu'on a gardé,
                    # on l'ignore car le caractère a déjà été "consommé" par la fusion
                    if idx >= last_idx + 2:
                        keep.append(idx)
                        last_idx = idx
                indices = np.array(keep)

            """
            # Gérer les chevauchements (ex: 'aaa' avec la règle ('a','a'))
            keep = np.ones(len(indices), dtype=bool)
            for j in range(len(indices) - 1):
                if indices[j+1] == indices[j] + 1:
                    keep[j+1] = False
            indices = indices[keep]
            """
            
            # Remplacement vectorisé
            data[indices] = new_id
            # Supprimer le deuxième élément de chaque paire fusionnée
            data = np.delete(data, indices + 1)

        # self.ids = data.tolist()
        self.enc = data.tolist()
#        self.stats['len_final'] = len(self.ids)
#        self.stats['unique_tokens'] = len(set(self.ids))
#        self.stats['compression_ratio'] = self.stats['len_init'] / self.stats['len_final']
        self.stats['temps_encodage'] = time.time() - start_time
            
        return data.tolist()

    def get_stats(self):
        """Retourne les statistiques"""
        return self.stats



# Pattern standard GPT-4 pour découper proprement (mots, chiffres, ponctuation)
GPT_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class OptimizedTokenizer:
    def __init__(self, merges):
        """
        merges: Votre dictionnaire actuel {(p0, p1): new_id}
        """
        self.merges = merges
        # On crée un dictionnaire de "rang" pour savoir quelle fusion est prioritaire
        # (On suppose que vos merges sont ordonnés par ordre d'apprentissage)
        self.ranks = dict(zip(merges.keys(), range(len(merges))))
        
        # Compilation du regex pour la vitesse
        GPT_EP  = r"""(?i:[lcdtmnsj]|qu)'|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.pat = rex.compile(GPT_EP)
        
        # Cache interne pour mémoriser les mots déjà vus
        self.cache = {} 
        self._vocab()

    def bpe(self, token_ids):
        """
        Applique le BPE sur une liste d'entiers représentant UN SEUL mot.
        C'est la version optimisée de votre boucle while.
        """
        # On travaille sur une copie liste (plus rapide que numpy pour de petits tableaux < 100 items)
        ids = list(token_ids)
        
        while len(ids) >= 2:
            # 1. Trouver toutes les paires adjacentes
            stats = {}
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                stats[pair] = i # On garde l'index
            
            # 2. Quelle est la paire avec le plus petit rang (la plus prioritaire) ?
            # On cherche si une des paires existe dans self.ranks
            best_pair = None
            min_rank = float('inf')
            
            for pair in stats:
                rank = self.ranks.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair
            
            # Si aucune paire n'est fusionnable, on arrête
            if best_pair is None:
                break
                
            # 3. Fusionner la meilleure paire
            # On remplace toutes les occurrences de best_pair par new_id
            new_id = self.merges[best_pair]
            i = 0
            new_ids = []
            while i < len(ids):
                # Si on n'est pas au dernier élément et qu'on trouve la paire
                if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i+1] == best_pair[1]:
                    new_ids.append(new_id)
                    i += 2 # On saute les deux éléments fusionnés
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            
        return ids

    def encode(self, text):
        """
        Encode un texte complet en utilisant le découpage + cache.
        """
        tokens = []
        # 1. Découpage du texte en morceaux (mots) via Regex
        chunks = self.pat.findall(text)
        
        for chunk in chunks:
            # 2. Conversion en bytes pour avoir les IDs de base (UTF-8)
            chunk_bytes = chunk.encode("utf-8")
            
            # 3. Vérification du cache
            # On utilise le tuple d'octets comme clé (hashable)
            if chunk_bytes in self.cache:
                tokens.extend(self.cache[chunk_bytes])
            else:
                # 4. Calcul BPE si mot inconnu
                # Conversion des bytes en liste d'entiers
                ids = list(chunk_bytes)
                merged_ids = self.bpe(ids)
                
                # Mise en cache
                self.cache[chunk_bytes] = merged_ids
                tokens.extend(merged_ids)
                
        return tokens
    
    def decode(self, array_=None, vocab=None): # Décode une liste de tokens en texte
        if (vocab==None):
            vocab_ = self.vocab
        else:
            vocab_ = vocab
        if (array_==None):
            array = self.get_array(self.ids)
        else:
            array = array_
        decoded_bytes = bytearray()
        for idx in array:
            decoded_bytes.extend(vocab_[idx])
        return decoded_bytes.decode('utf-8', errors='replace')
    
    def _vocab(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        # self._compute_pattern()
    
