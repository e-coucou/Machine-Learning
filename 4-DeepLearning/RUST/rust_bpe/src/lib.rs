use pyo3::prelude::*;
//use regex::Regex;
use fancy_regex::Regex; // On utilise fancy_regex maintenant
use std::collections::HashMap;
use candle_core::{Device, DType, Tensor};

use std::fs::File;
use std::fs;
use std::io::BufWriter;
use std::io::{self, Write};
use rayon::prelude::*;

#[pyclass]
pub struct BPETrainer {
    vocab_size_target: usize,
    re: Regex,
	#[pyo3(get)]
    merges: HashMap<(i32, i32), i32>, 
    #[pyo3(get)]
    vocab: HashMap<i32, Vec<u8>>,
}


#[pymethods]
impl BPETrainer {
    #[new]
    fn new(target_extra: usize) -> Self {
        let pattern_ep  = r"(?i:[lcdtmnsj]|qu)'|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
        BPETrainer {
            vocab_size_target: 256 + target_extra,
            re: Regex::new(pattern_ep).unwrap(),
			merges: HashMap::new(),
            vocab: HashMap::new(),        
        }
    }

    fn train(&mut self, text: String) -> PyResult<(HashMap<i32, Vec<u8>>, Vec<i32>)> {
        // 1. Initialisation du vocabulaire
        self.vocab = (0..256)
            .map(|i| (i as i32, vec![i as u8]))
            .collect();

        // 2. Création du tableau "plat" (Flat Array)
        // On utilise -1 (i32::MAX) comme séparateur de mot
        const SEP: i32 = i32::MAX;
        let mut flat_data: Vec<i32> = Vec::with_capacity(text.len());

		for m in self.re.find_iter(&text) {
		    // fancy_regex::find_iter renvoie un Result<Match, Error>
		    if let Ok(mat) = m {
		        for &byte in mat.as_str().as_bytes() {
		            flat_data.push(byte as i32);
		        }
		        flat_data.push(SEP); 
		    }
		}

        let mut current_vocab_size = 256;

        while current_vocab_size < self.vocab_size_target {
            // ÉTAPE A : Compter les paires (sans franchir le séparateur)
            let mut counts = HashMap::new();
            for window in flat_data.windows(2) {
                if window[0] != SEP && window[1] != SEP {
                    let pair = (window[0], window[1]);
                    *counts.entry(pair).or_insert(0) += 1;
                }
            }
            
			// Version 2 en parallele ... mais plus lente
			/*
			let counts = flat_data
		        .par_windows(2) // Fenêtres de 2 en parallèle
		        .filter(|w| w[0] != SEP && w[1] != SEP) // On ignore les séparateurs
		        .fold(HashMap::new, |mut acc, w| {
		            *acc.entry((w[0], w[1])).or_insert(0) += 1;
		            acc
		        })
		        .reduce(HashMap::new, |mut a, b| {
		            for (k, v) in b { *a.entry(k).or_insert(0) += v; }
		            a
		        });
		    */

            if let Some(((p1, p2), _)) = counts.into_iter().max_by_key(|&(_, count)| count) {
                let new_id = current_vocab_size as i32;

				self.merges.insert((p1, p2), new_id);

                // Mise à jour du vocabulaire
                let mut new_bytes = self.vocab.get(&p1).unwrap().clone();
                new_bytes.extend(self.vocab.get(&p2).unwrap());
                self.vocab.insert(new_id, new_bytes);

                // ÉTAPE B : Fusion in-place (très rapide)
                let mut write_idx = 0;
                let mut read_idx = 0;

				while read_idx < flat_data.len() {
				    if read_idx < flat_data.len() - 1 
				       && flat_data[read_idx] == p1 
				       && flat_data[read_idx + 1] == p2 
				       && flat_data[read_idx] != SEP // Sécurité supplémentaire
				    {
				        flat_data[write_idx] = new_id;
				        read_idx += 2;
				    } else {
				        flat_data[write_idx] = flat_data[read_idx];
				        read_idx += 1;
				    }
				    write_idx += 1;
				}
				flat_data.truncate(write_idx); // on ajuste après fusion
                
                current_vocab_size += 1;
            } else {
                break;
            }
        }

        // On enlève les séparateurs pour le résultat final
        let final_ids: Vec<i32> = flat_data.into_iter()
            .filter(|&x| x != SEP)
            .collect();

        Ok((self.vocab.clone(), final_ids))
    } // end train

    fn encode(&self, text: String) -> Vec<i32> {
        // 1. Découpage initial via Regex (comme au début du train)
        let mut words: Vec<Vec<i32>> = self.re.find_iter(&text)
        	.par_bridge() // en parallele
            .filter_map(|m| m.ok())
            .map(|m| m.as_str().as_bytes().iter().map(|&b| b as i32).collect())
            .collect();

        // 2. Application des merges sur chaque mot
        words.par_iter_mut().for_each(|word| {
            while word.len() >= 2 {
                // On cherche la paire dans le mot qui a le plus petit ID de fusion (priorité)
                let mut best_pair = None;
                let mut best_rank = i32::MAX;

                for window in word.windows(2) {
                    let pair = (window[0], window[1]);
                    if let Some(&new_id) = self.merges.get(&pair) {
                        if new_id < best_rank {
                            best_rank = new_id;
                            best_pair = Some(pair);
                        }
                    }
                }

                // Si on a trouvé une paire fusionnable, on l'applique
                if let Some((p1, p2)) = best_pair {
                    let mut new_word = Vec::with_capacity(word.len() - 1);
                    let mut i = 0;
                    while i < word.len() {
                        if i < word.len() - 1 && word[i] == p1 && word[i+1] == p2 {
                            new_word.push(best_rank);
                            i += 2;
                        } else {
                            new_word.push(word[i]);
                            i += 1;
                        }
                    }
                    *word = new_word;
                } else {
                    break; // Plus rien à fusionner dans ce mot
                }
            }
        });

        words.into_iter().flatten().collect()
    } // encoder

    fn save(&self, path: String) -> PyResult<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        
        // On crée une structure simple pour le JSON
        let data = serde_json::json!({
            "vocab": self.vocab,
            "merges": self.merges.iter().map(|((p1, p2), res)| vec![p1, p2, res]).collect::<Vec<_>>()
        });

        serde_json::to_writer_pretty(writer, &data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
        })?;
        
        Ok(())
    } //en save

	fn decode(&self, ids: Vec<i32>) -> String {
        let mut merged_bytes = Vec::new();

        for id in ids {
            if let Some(bytes) = self.vocab.get(&id) {
                // On ajoute les octets du token à notre liste finale
                merged_bytes.extend(bytes);
            }
        }

        // On transforme la liste d'octets en String lisible
        // "lossy" remplace les caractères invalides par  au lieu de crasher
        String::from_utf8_lossy(&merged_bytes).into_owned()
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        // 1. Lire le fichier
        let content = fs::read_to_string(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        // 2. Parser le JSON
        let data: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // 3. Reconstruire le Vocab (ID -> Bytes)
        let mut vocab = HashMap::new();
        if let Some(obj) = data["vocab"].as_object() {
            for (id_str, bytes_val) in obj {
                let id = id_str.parse::<i32>().unwrap();
                let bytes: Vec<u8> = serde_json::from_value(bytes_val.clone()).unwrap();
                vocab.insert(id, bytes);
            }
        }

        // 4. Reconstruire les Merges ((P1, P2) -> NewID)
        let mut merges = HashMap::new();
        if let Some(arr) = data["merges"].as_array() {
            for entry in arr {
                if let Some(m) = entry.as_array() {
                    // On a stocké [p1, p2, res]
                    let p1 = m[0].as_i64().unwrap() as i32;
                    let p2 = m[1].as_i64().unwrap() as i32;
                    let res = m[2].as_i64().unwrap() as i32;
                    merges.insert((p1, p2), res);
                }
            }
        }

        // 5. Retourner l'instance
        let pattern_ep  = r"(?i:[lcdtmnsj]|qu)'|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
        Ok(BPETrainer {
            vocab_size_target: vocab.len(),
            re: Regex::new(pattern_ep).unwrap(),
            merges,
            vocab,
        })
    } // end load

} // end BPETrainer


/// A Python module implemented in Rust.
#[pymodule]
fn rust_bpe (_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<BPETrainer>()?;
	Ok(())
}

