import sys
from ase.io import read, write

if len(sys.argv) != 3:
    print("Uso: python traj_to_xyz.py input.traj output.xyz")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Leer una estructura cada 500 frames
atoms = read(input_file, index=":")

write(output_file, atoms, format="xyz")

print(f"Convertidas {len(atoms)} estructuras a {output_file}")
