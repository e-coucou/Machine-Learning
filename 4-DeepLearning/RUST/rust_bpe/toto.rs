use candle_core::{Device, DType, Tensor};
use pyo3::prelude::*;
use std::collections::HashMap;


use std::io::{self, Write};

#[pyclass]
struct BPETrainer{
	vocab_size_target: usize,
}

#[pymethods]
impl BPETrainer {
	#[new]
	fn new(target_extra: usize, chars: usize) -> Self {
		BPETrainer {
			vocab_size_target: chars +target_extra,
		}
	}
	fn train(&self, text: String) -> PyResult<(HashMap<Vec<u8>, i32>, Vec<i32>, HashMap<i32, Vec<u8>>)> {
		let mut tokens: Vec<u8> = text.into_bytes();
		let mut ids: Vec<i32> = tokens.iter().map(|&b | b as i32).collect();

		let mut vocab: HashMap<i32, Vec<u8>> = (0..256)
			.map(|i| (i as i32, vec![i as u8]))
			.collect();

		let mut current_vocab_size = 256;

		while current_vocab_size < self.vocab_size_target {
//			println!("Les compteurs : {:?} {:?}", current_vocab_size, self.vocab_size_target);
//			io::stdout().flush().unwrap();
			let mut paires = HashMap::new();
			for w in ids.windows(2) {
				let paire = (w[0],w[1]);
				*paires.entry(paire).or_insert(0) += 1;
			}
			let best = paires.into_iter().max_by_key(|&(_, count) | count);

			if let Some( ((p1,p2), _)) = best {
				let new_id = current_vocab_size as i32;
				let mut new_paire = vocab.get(&p1).unwrap().clone();
				new_paire.extend(vocab.get(&p1).unwrap());
				vocab.insert(new_id, new_paire);
				let mut new_ids = Vec::with_capacity(ids.len());
				let mut i = 0;
				while i < ids.len() {
					if i <ids.len()-1 && ids[i] == p1 && ids[i+1] == p2 {
						new_ids.push(new_id);
						i += 2;
					} else { 
						new_ids.push(ids[i]);
						i += 1;
					}
				}
				ids = new_ids;
				current_vocab_size += 1
			} else {
				break;
			}
		}
		let encoder: HashMap<Vec<u8>, i32> = vocab.iter()
			.map(|(id, bytes)| (bytes.clone(), *id))
			.collect();

		Ok((encoder, ids, vocab))
	}
}
/// A Python module implemented in Rust.
#[pymodule]
fn rust_bpe (_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<BPETrainer>()?;
	Ok(())
}
