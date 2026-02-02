import test_python as rust

print("--- Test de communication ---")
resultat = rust.addition_rust(10, 32)
print(f"Le résultat calculé par Rust est : {resultat}")