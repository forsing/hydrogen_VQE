import csv
from pathlib import Path

import pennylane as qml
from pennylane import numpy as np

# Loto 7/39 — VQE stil (kao polazni hydrogen) + qc25 (5×5=25 qubita)
# 5 brojeva iz kola, 6. i 7. izvedeni; opsezi po poziciji.
NUM_QUBITS_PER_POS = 5
NUM_POSITIONS = 5  # prvih 5 brojeva u kolu
LOTTO_WIRES = NUM_QUBITS_PER_POS * NUM_POSITIONS  # 25
LOTTO_SEED = 39
LOTTO_VQE_STEPS = 80
LOTTO_VQE_LR = 0.05
LOTTO_LAYERS = 2
LOTTO_MF_SCALE = 0.35
LOTTO_H_SCALE = 0.08
LOTTO_SHOTS = 4646
MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
LOTTO_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4646_k54.csv"


def load_all_draws(csv_path=LOTTO_CSV_PATH):
    """Učitaj sva validna kola iz celog CSV-a (sortirana)."""
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:7])
            except ValueError:
                continue
            if all(1 <= x <= 39 for x in draw) and len(set(draw)) == 7:
                draws.append(draw)
    if not draws:
        raise ValueError(f"No valid 7/39 draw found in {csv_path}")
    return np.array(draws, dtype=int)


def encode_value_bits(value, n_q=NUM_QUBITS_PER_POS):
    """Broj -> n_q bitova (kao qc25 encode_position)."""
    v = int(value)
    return [(v >> i) & 1 for i in range(n_q)]


def find_unique(start_val, used_set, idx):
    mv, mv_max = MIN_VAL[idx], MAX_VAL[idx]
    rng = mv_max - mv + 1
    v = ((int(start_val) - mv) % rng) + mv
    tries = 0
    while v in used_set and tries < rng:
        v = mv + ((v - mv + 1) % rng)
        tries += 1
    if v in used_set:
        for cand in range(mv, mv_max + 1):
            if cand not in used_set:
                return int(cand)
    return int(v)


def bitstring_to_loto_with_7(bitstring_bits):
    """
    qc25 dekodovanje: 25 bitova -> 5 brojeva po opsezima,
    6. i 7. izvedeni. Generiše kombinaciju (nije CSV lookup).
    """
    main_numbers = []
    for pos in range(NUM_POSITIONS):
        start = pos * NUM_QUBITS_PER_POS
        chunk = bitstring_bits[start : start + NUM_QUBITS_PER_POS]
        val = 0
        for i, b in enumerate(chunk):
            val |= (int(b) & 1) << i
        mv, mv_max = MIN_VAL[pos], MAX_VAL[pos]
        mapped = (val % (mv_max - mv + 1)) + mv
        main_numbers.append(int(mapped))

    sum_main = sum(main_numbers)
    start6 = (sum_main) % (MAX_VAL[5] - MIN_VAL[5] + 1) + MIN_VAL[5]
    sixth = find_unique(start6, set(main_numbers), 5)
    used = set(main_numbers) | {sixth}
    start7 = (sum_main + sixth) % (MAX_VAL[6] - MIN_VAL[6] + 1) + MIN_VAL[6]
    seventh = find_unique(start7, used, 6)
    return sorted(main_numbers + [sixth, seventh])


def block_bit_matrix(draws, pos):
    """Bitovi jedne pozicije iz celog CSV: shape (N, 5)."""
    return np.array([encode_value_bits(int(d[pos])) for d in draws], dtype=float)


def neighbor_mean_field_h(draws, pos):
    """
    Cross-position mean-field iz CSV:
    za svaki lokalni bit a, efektivno polje od susednih pozicija
    sum_b J_ab * <Z_b>_susjed, gde je J_ab korelacija bitova iz istorije.
    Trening ostaje 5-qubit (bez OOM).
    """
    local = block_bit_matrix(draws, pos)
    extra_h = np.zeros(NUM_QUBITS_PER_POS, dtype=float)

    for nbr in (pos - 1, pos + 1):
        if nbr < 0 or nbr >= NUM_POSITIONS:
            continue
        nbr_bits = block_bit_matrix(draws, nbr)
        nbr_z_mean = 2.0 * nbr_bits.mean(axis=0) - 1.0
        for a in range(NUM_QUBITS_PER_POS):
            for b in range(NUM_QUBITS_PER_POS):
                corr = float(np.corrcoef(local[:, a], nbr_bits[:, b])[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                # Mean-field za -J Z_a Z_b  ->  dodatni ugal za Z_a: -J <Z_b>
                extra_h[a] += LOTTO_MF_SCALE * corr * float(nbr_z_mean[b])
    return extra_h


def build_block_hamiltonian(draws, pos):
    """
    Ising Hamiltonijan jednog 5-qubit bloka iz celog CSV (po poziciji)
    + mean-field kuplovanje sa susednim pozicijama.
    Lokalno 2^5 — sprečava OOM od 2^25 statevectora.
    """
    bits = block_bit_matrix(draws, pos)
    extra_h = neighbor_mean_field_h(draws, pos)

    coeffs = []
    ops = []
    for i in range(NUM_QUBITS_PER_POS):
        h_i = 2.0 * float(bits[:, i].mean()) - 1.0
        coeffs.append(-LOTTO_H_SCALE * (h_i + float(extra_h[i])))
        ops.append(qml.PauliZ(i))

    for a in range(NUM_QUBITS_PER_POS):
        for b in range(a + 1, NUM_QUBITS_PER_POS):
            corr = float(np.corrcoef(bits[:, a], bits[:, b])[0, 1])
            if np.isnan(corr):
                corr = 0.0
            coeffs.append(-LOTTO_H_SCALE * corr)
            ops.append(qml.PauliZ(a) @ qml.PauliZ(b))

    return qml.Hamiltonian(coeffs, ops)


def build_start_bits_for_pos(draws, pos):
    """Referentno stanje bloka (analog hf_state): poslednje kolo, ta pozicija."""
    return encode_value_bits(int(draws[-1][pos]))


def train_block_vqe(draws, pos):
    """VQE na jednom 5-qubit bloku — puni parametri (layers × 5 × 3)."""
    H_block = build_block_hamiltonian(draws, pos)
    start_bits = build_start_bits_for_pos(draws, pos)
    wires = NUM_QUBITS_PER_POS

    np.random.seed(LOTTO_SEED + pos)
    weights = np.array(
        0.01 * np.random.randn(LOTTO_LAYERS, wires, 3),
        requires_grad=True,
    )

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def energy(w):
        qml.BasisState(np.array(start_bits), wires=range(wires))
        qml.StronglyEntanglingLayers(weights=w, wires=range(wires))
        return qml.expval(H_block)

    opt = qml.GradientDescentOptimizer(stepsize=LOTTO_VQE_LR)
    loss_history = []
    for _ in range(LOTTO_VQE_STEPS):
        weights = opt.step(energy, weights)
        loss_history.append(float(energy(weights)))

    return weights, start_bits, loss_history[-1]


def sample_block_bits(weights, start_bits, shots):
    """Sample 5 bitova jednog treniniranog bloka."""
    wires = NUM_QUBITS_PER_POS
    dev = qml.device("default.qubit", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev)
    def sample_bits(w):
        qml.BasisState(np.array(start_bits), wires=range(wires))
        qml.StronglyEntanglingLayers(weights=w, wires=range(wires))
        return qml.sample(wires=range(wires))

    return sample_bits(weights)


def combo_from_block_samples(block_samples_list):
    """
    Predikcija: spoji 5×5 bit uzorke u 25-bit qc25 string,
    dekoduj u 7 brojeva. Najčešća generisana kombinacija.
    """
    counts = {}
    n = len(block_samples_list[0])
    for i in range(n):
        bits25 = []
        for pos in range(NUM_POSITIONS):
            bits25.extend(int(b) for b in block_samples_list[pos][i])
        combo = tuple(bitstring_to_loto_with_7(bits25))
        counts[combo] = counts.get(combo, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return list(ranked[0][0])


def run_loto_vqe(draws):
    """
    qc25 VQE bez OOM:
    - trenira 5 odvojenih 5-qubit VQE blokova (ceo CSV)
    - sample-uje sve blokove, spaja u 25 bitova
    - dekoduje u 7/39 (ceo prostor preko qc25 + izvedeni 6/7)
    """
    all_weights = []
    all_starts = []
    energies = []
    n_params = 0

    for pos in range(NUM_POSITIONS):
        weights, start_bits, e_final = train_block_vqe(draws, pos)
        all_weights.append(weights)
        all_starts.append(start_bits)
        energies.append(e_final)
        n_params += int(weights.size)

    block_samples = [
        sample_block_bits(all_weights[pos], all_starts[pos], LOTTO_SHOTS)
        for pos in range(NUM_POSITIONS)
    ]
    combo = combo_from_block_samples(block_samples)
    mean_energy = float(np.mean(energies))
    return combo, mean_energy, mean_energy, n_params


def run_loto_prediction_pipeline():
    draws = load_all_draws()
    combo, final_energy, ground_energy, n_params = run_loto_vqe(draws)
    print("Loto next-combo prediction:", combo)
    print(f"Loto VQE final energy: {final_energy:.8f}")
    print(f"Loto VQE ground energy: {ground_energy:.8f}")
    print(f"Loto VQE qubits: {LOTTO_WIRES} (qc25 = 5x5 blocks), params: {n_params}")


if __name__ == "__main__":
    run_loto_prediction_pipeline()


"""
Loto next-combo prediction: [4, x, 9, y, 24, z, 36]
Loto VQE final energy: 0.03338452
Loto VQE ground energy: 0.03338452
Loto VQE qubits: 25 (qc25 = 5x5 blocks), params: 150
"""
