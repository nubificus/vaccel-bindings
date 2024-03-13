use env_logger::Env;
use log::info;
use std::path::PathBuf;

use vaccel::torch;
use vaccel::Session;

use std::collections::HashMap;
use std::io::{self, Write};

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use std::time::Instant;

extern crate utilities;

fn vaccel_preprocess(
    text: &str,
    token2id: &HashMap<String, i32>,
    max_length: usize,
    log: bool,
) -> (Vec<Vec<i32>>, Vec<Vec<Vec<i32>>>) {
    let pad_token = "[PAD]";
    let start_token = "[CLS]";
    let end_token = "[SEP]";
    let pad_token_id = *token2id.get(pad_token).expect("PAD token not found");
    let start_token_id = *token2id.get(start_token).expect("CLS token not found");
    let end_token_id = *token2id.get(end_token).expect("SEP token not found");

    let mut input_ids = vec![pad_token_id; max_length];
    let mut masks = vec![0; max_length];
    input_ids[0] = start_token_id;
    masks[0] = 1;

    let mut input_id = 1;
    for word in text.split_whitespace() {
        if input_id >= max_length - 1 {
            break;
        }
        let word_id = *token2id.get(word).unwrap_or(&pad_token_id);
        masks[input_id] = 1;
        input_ids[input_id] = word_id;
        if log {
            println!("{} : {}", word, word_id);
        }
        input_id += 1;
    }

    masks[input_id] = 1;
    input_ids[input_id] = end_token_id;

    if log {
        io::stdout().write_all(b"Input IDs: ").unwrap();
        for i in &input_ids {
            print!("{} ", i);
        }
        println!();

        io::stdout().write_all(b"Masks: ").unwrap();
        for i in &masks {
            print!("{} ", i);
        }
        println!();
    }

    let input_ids_tensor = vec![input_ids];
    let masks_tensor = vec![vec![masks]];
    
    (input_ids_tensor, masks_tensor)
}

fn load_vocab<P: AsRef<Path>>(
    vocab_path: P,
) -> io::Result<(HashMap<String, i32>, HashMap<i32, String>)> {
    let file = File::open(vocab_path)?;
    let reader = BufReader::new(file);

    let mut token2id = HashMap::new();
    let mut id2token = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            continue; // Skip malformed lines
        }
        let token = parts[0].to_string();
        let token_id = parts[1].parse::<i32>().unwrap_or(-1); // Consider better error handling here

        token2id.insert(token.clone(), token_id);
        id2token.insert(token_id, token);
    }

    Ok((token2id, id2token))
}

fn convert_to_f32_slice(nested_vec: Vec<Vec<Vec<i32>>>) -> Vec<f32> {
    nested_vec
        .into_iter()  
        .flatten()    
        .flatten()   
        .map(|x| x as f32)  
        .collect()    
}

fn main() -> utilities::Result<()> {

    let vocab_path = "/home/ilias/vaccel-torch-bert-example/bert_cased_vocab.txt";

    match load_vocab(vocab_path) {
        Ok((token2id, id2token)) => {
            println!("Vocabulary loaded successfully!");

            let text = "hello world";
            let max_length = 32;
            let log = true;
            let (input_ids, masks) = vaccel_preprocess(text, &token2id, max_length, log);

            println!("Preprocessed input:");
            println!("{:?}", input_ids);
            println!("Preprocessed masks:");
            println!("{:?}", masks);

            env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();
            let mut sess = Session::new(0)?;
            info!("New session {}", sess.id());

            let path = PathBuf::from("/home/ilias/model2");
            let mut model = torch::SavedModel::new().from_export_dir(&path)?;
            info!("New saved model from export dir: {}", model.id());

            sess.register(&mut model)?;
            info!("Registered model {} with session {}", model.id(), sess.id());

            let run_options = torch::Buffer::new(&[]);

            let input_ids_f32: Vec<f32> = input_ids.into_iter()
                .flat_map(|inner_vec| inner_vec.into_iter().map(|x| x as f32))
                .collect();
            let input_ids_data: &[f32] = &input_ids_f32;

            let flat_f32_vec = convert_to_f32_slice(masks);
            let masks_data: &[f32] = &flat_f32_vec;

            let in_tensor1 = torch::Tensor::<f32>::new(&[1,32]).with_data(input_ids_data)?;
            let in_tensor2 = torch::Tensor::<f32>::new(&[1,1,32]).with_data(masks_data)?;
            
            let mut sess_args = torch::TorchArgs::new();
            let mut jitload = torch::TorchJitLoadForward::new();

            sess_args.set_run_options(&run_options);
            sess_args.add_input(&in_tensor1);
            sess_args.add_input(&in_tensor2);

            let mut result = jitload.jitload_forward(&mut sess, &mut sess_args, &mut model)?;
            
            for num in 0..100 {
                let now = Instant::now();
                result = jitload.jitload_forward(&mut sess, &mut sess_args, &mut model)?;
                println!("Plugin-Call Time: {}", now.elapsed().as_millis());
            }
            

            match result.get_output::<f32>(0) {
                Ok(out) => {
                    println!("Success");
                    println!(
                        "Output tensor => type:{:?} nr_dims:{}",
                        out.data_type(),
                        out.nr_dims()
                    );
                    for i in 0..out.nr_dims() {
                        println!("dim[{}]: {}", i, out.dim(i as usize).unwrap());
                    }
                }
                Err(err) => println!("Torch JitLoadForward failed: '{}'", err),
            }
        
            sess.close()?;
        }
        Err(e) => {
            println!("Failed to load the vocabulary: {}", e);
            std::process::exit(1); 
        }
    }
    
    std::process::exit(1); 
}
