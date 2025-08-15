import torch as T
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def sample_cancer_expressions(n_cells, marker_genes, n_genes, cell_library_size):
    gene_expression_probs = T.ones(n_genes, dtype=T.float32)
    gene_expression_probs[marker_genes] = 10.
    N = n_cells * cell_library_size
    gene_dist = T.distributions.Multinomial(total_count=N, probs=gene_expression_probs)
    gene_counts = gene_dist.sample()
    return gene_counts


def sample_healthy_expressions(n_cells, n_genes, cell_library_size):
    gene_expression_probs = T.ones(n_genes, dtype=T.float32)
    N = n_cells * cell_library_size
    gene_dist = T.distributions.Multinomial(total_count=N, probs=gene_expression_probs)
    gene_counts = gene_dist.sample()
    return gene_counts


def generate_data(cells_per_spot, n_genes, cell_library_size, marker_genes, n_spots, boarder_size=18):
    boarders = T.tensor([30., 60, 80., 40., 20.], dtype=T.float32)
    J = len(boarders)
    spot_coordinates = T.arange(n_spots, dtype=T.float32)

    true_distributions = T.zeros((n_spots, J, n_genes))
    noisy_distributions = T.zeros((n_spots, J, n_genes))
    offsets = T.zeros((n_spots, J))

    for j in range(J):
        for i, x in enumerate(spot_coordinates):
            offsets[i, j] = x - boarders[j]
            if x < boarders[j] - boarder_size / 2:
                gene_counts = sample_cancer_expressions(cells_per_spot, marker_genes, n_genes, cell_library_size)
                true_distributions[i, j, :] = gene_counts
                noisy_distributions[i, j, :] = gene_counts

            elif x > boarders[j] + boarder_size / 2:
                gene_counts = sample_healthy_expressions(cells_per_spot, n_genes, cell_library_size)
                true_distributions[i, j, :] = gene_counts
                noisy_distributions[i, j, :] = gene_counts

            else:
                power = T.abs(boarders[j] - x) / 10 + 0.09
                prob = 0.99 ** power * 0.5 ** (1 - power)
                # the closer the spot is to the boundary, the closer to uniform distribution of cell types
                probs = T.tensor([prob, 1 - prob], dtype=T.float32)
                if x >= boarders[j] - boarder_size / 2 and x <= boarders[j]:
                    # generate true data using varying probabilities
                    n_cancer_cells, n_healthy_cells = T.distributions.Multinomial(cells_per_spot, probs).sample().to(dtype=T.int)

                else:
                    assert  x <= boarders[j] + boarder_size / 2 and x > boarders[j]
                    n_healthy_cells, n_cancer_cells = T.distributions.Multinomial(cells_per_spot, probs).sample().to(dtype=T.int)

                # torch.distributions.Multinomial takes an integer (not tensor) as number of trials
                n_cancer_cells = n_cancer_cells.item()
                n_healthy_cells = n_healthy_cells.item()

                if n_healthy_cells > 0:
                    healthy_gene_counts = sample_healthy_expressions(n_healthy_cells, n_genes, cell_library_size)
                    true_distributions[i, j, :] += healthy_gene_counts
                if n_cancer_cells > 0:
                    cancer_gene_counts = sample_cancer_expressions(n_cancer_cells, marker_genes, n_genes,
                                                                   cell_library_size)
                    true_distributions[i, j, :] += cancer_gene_counts

                # generate noisy estimates, incapable of distinguishing cancer and healthy cells in the boundary region
                n_cancer_cells, n_healthy_cells = T.distributions.Multinomial(cells_per_spot, T.tensor([0.5, 0.5])).sample().to(dtype=T.int)
                n_cancer_cells = n_cancer_cells.item()
                n_healthy_cells = n_healthy_cells.item()

                if n_healthy_cells > 0:
                    healthy_gene_counts = sample_healthy_expressions(n_healthy_cells, n_genes, cell_library_size)
                    noisy_distributions[i, j, :] += healthy_gene_counts
                if n_cancer_cells > 0:
                    cancer_gene_counts = sample_cancer_expressions(n_cancer_cells, marker_genes, n_genes,
                                                                   cell_library_size)
                    noisy_distributions[i, j, :] += cancer_gene_counts

    # CP10K normalization and log1p transformed data
    true_distributions = T.log(10000 * true_distributions / true_distributions.sum(-1).unsqueeze(-1)  + 1)
    # true_distributions /= true_distributions.sum(0).unsqueeze(0)
    noisy_distributions = T.log(10000 * noisy_distributions / noisy_distributions.sum(-1).unsqueeze(-1) + 1)
    # noisy_distributions /= noisy_distributions.sum(0).unsqueeze(0)

    true_distributions, noisy_distributions = true_distributions[..., marker_genes], noisy_distributions[..., marker_genes]
    # true_distributions /= true_distributions.sum(1).unsqueeze(1)
    # noisy_distributions /= noisy_distributions.sum(1).unsqueeze(1)

    # return only marker gene data
    return true_distributions, noisy_distributions, spot_coordinates, boarders, offsets


def sample_data(bs, sampled_slides, dataset):
    true_distributions = dataset['true_distributions']  # (n_spots, n_slides, n_marker_genes)
    noisy_distributions = dataset['noisy_distributions']
    spot_coordinates = dataset['spot_coordinates']  # (n_spots)
    n_spots = spot_coordinates.shape[0]

    spot_ids = T.tensor(np.random.choice(n_spots, bs, replace=True), dtype=T.int)
    c = spot_coordinates[spot_ids] / n_spots
    x0 = true_distributions[spot_ids, 0]   # + T.randn((bs, true_distributions.shape[-1]), dtype=T.float32) * 0.00001
    x1 = true_distributions[spot_ids, -1]  # + T.randn((bs, true_distributions.shape[-1]), dtype=T.float32) * 0.00001
    xhat_t = noisy_distributions[spot_ids, sampled_slides] # + T.randn((bs, true_distributions.shape[-1]), dtype=T.float32) * 0.01

    return x0, x1, xhat_t, c



if __name__ == '__main__':
    n_genes = 10000
    cell_library_size = 3000
    n_cells = 10
    n_marker_genes = 15
    for i in tqdm(range(5)):
        marker_genes = T.tensor(np.random.choice(np.arange(n_genes), n_marker_genes, replace=False), dtype=T.int)
        true_distributions, noisy_distributions, spot_coordinates, boarders, offsets = generate_data(10,
                                                                                                     n_genes,
                                                                                            cell_library_size,
                                                                                            marker_genes, n_spots=100,
                                                                                            boarder_size=18)
        output = {
            'true_distributions': true_distributions,
            'noisy_distributions': noisy_distributions,
            'spot_coordinates': spot_coordinates,
            'boarders': boarders,
            'offsets': offsets,
        }
        save_path = f'../data/boundaries/dataset_{i}.pt'
        T.save(output, save_path)

