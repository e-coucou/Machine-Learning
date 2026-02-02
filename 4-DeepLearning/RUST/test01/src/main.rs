use std::io;

fn main() {
    println!("🦀 Salut ! Je suis un programme écrit en Rust sur Mac M1.");
    println!("Comment t'appelles-tu ?");

    let mut nom = String::new();

    io::stdin()
        .read_line(&mut nom)
        .expect("Échec de la lecture de l'entrée");

    println!("Enchanté, {} ! Ton environnement Rust est prêt.", nom.trim());
}