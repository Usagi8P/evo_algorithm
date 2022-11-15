from math import exp
import numpy as np
import random
from typing import Optional

class Commuinity():
    """A comunity contains all individuals"""
    def __init__(self,community_size: int, n_inputs: int = 0, n_outputs: int = 0, additional_genes: int = 0) -> None:
        self.individuals: list[Individual] = []
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.additional_genes = additional_genes
        self.genome_size = n_inputs + n_outputs + additional_genes
        self._create_community(community_size, self.n_inputs, self.n_outputs, self.additional_genes)

    def _create_community(self,community_size: int, n_inputs: int, n_outputs: int, additional_genes: int) -> None:
        for i in range(community_size):
            self.individuals.append(Individual(i,  n_inputs, n_outputs, additional_genes))

    def add_inputs(self, n_inputs: int) -> None:
        self.n_inputs += n_inputs
        for individual in self.individuals:
            individual.add_inputs(n_inputs)

    def add_genes(self, additional_genes: int, activation_function: str = 'linear') -> None:
        self.additional_genes += additional_genes
        for individual in self.individuals:
            individual.add_genes(additional_genes, activation_function)

    def add_outputs(self, n_outputs: int) -> None:
        self.n_outputs += n_outputs
        for individual in self.individuals:
            individual.add_outputs(n_outputs)

    def show_community(self) -> None:
        for individual in self.individuals:
            print('Individual ID: ', individual.id)
            print('  Genome Size: ', individual.genome_size)
            for gene in individual.genome:
                print('    Gene ID: ', gene.id)
                print('    Activation function: ', gene.activation_function)
                print('    Output ID: ', gene.output_id)
                print('    Bias: ', gene.bias)
                print('')

    def fit(self, X_train, y_train, n_generations) -> list[list[float]]:
        individual_outputs: list[list[float]] = []
        for individual in self.individuals:
            individual_outputs.append(individual.calculate_output(X_train))

        return individual_outputs

class Individual:
    """An individual is a single actor with a unique genome made of multiple genes"""
    def __init__(self,id: int, n_inputs: int, n_outputs: int, additional_genes: int) -> None:
        self.id = id
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.additional_genes = additional_genes
        self.genome_size = n_inputs + n_outputs + additional_genes
        self.input_ids: list[int] = list(range(n_inputs))
        self.output_ids: list[int] = list(range(n_outputs))
        self.genome: list[Gene] = []

        if self.genome_size > 0:
            self._create_genome()


    def add_inputs(self, n_inputs) -> None:
        for _ in range(n_inputs):
            id: int = len(self.genome)
            output = random.randint(0,self.genome_size-1)
            while output in self.input_ids + [id]:
                output = random.randint(0,self.genome_size-1)
            bias_weight = random.random()
            self.input_ids.append(id)
            self.genome.append(Gene(id,output, bias_weight))
        
    def add_genes(self, additional_genes: int, activation_function: str) -> None:
        output = random.randint(0,self.genome_size-1)
        while output in self.input_ids + [id]:
            output = random.randint(0,self.genome_size-1)
        bias_weight = random.random()
        for _ in range(additional_genes):
            self.genome.append(Gene(len(self.genome), output, bias_weight, activation_function))

    def add_outputs(self, n_outputs: int) -> None:
        for _ in range(n_outputs):
            id: int = len(self.genome)
            self.output_ids.append(id)
            self.genome.append(Gene(id, None, 1.0))

    def _create_genome(self) -> None:
        self.add_inputs(self.n_inputs)
        self.add_genes(self.additional_genes, activation_function='linear')
        self.add_outputs(self.n_outputs)

    def calculate_output(self, X_train: list[float]) -> list[float]:
        # Make list of inputs and outputs
        inputs: list[int] = []
        outputs: list[Optional[int]] = []
        for gene in self.genome:
            if gene.output_id is not None:
                inputs.append(gene.output_id)
                outputs.append(gene.output_id)

        # Check if any connection connects to the inputs
        if not any((match := input) in inputs for input in self.input_ids):
            return [np.nan]
        # Check if any connection connects to the outputs
        if not any((match := output) in outputs for output in self.output_ids):
            return [np.nan]

        genes_to_check: list[int] = []
        for gene_id in self.input_ids:
            active_gene = self.genome[gene_id]
            active_gene.input = X_train.pop()
            active_gene.calculate_gene_output()
            if active_gene.output_id is not None:
                self.genome[active_gene.output_id].input = active_gene.output
                genes_to_check.append(active_gene.output_id)

        i: int = 0
        while len(genes_to_check) > 0 and i < 50:
            active_gene = self.genome[genes_to_check[0]]
            active_gene.calculate_gene_output()
            if active_gene.output_id is not None:
                self.genome[active_gene.output_id].input = active_gene.output
                genes_to_check.append(active_gene.output_id)
            del genes_to_check[0]
            i += 1

        output_values: list[float] = []
        for gene_id in self.output_ids:
            active_gene = self.genome[gene_id]
            output_values.append(active_gene.output)

        return output_values

class Gene:
    """Each gene is a single node with any number of inputs an output and a bias"""
    def __init__(self, id: int, output_id: Optional[int], bias_weight: float, activation_function: str = 'linear') -> None:
        self.id = id
        self.activation_function = activation_function
        self.input: float = 0.0
        self.output: float = np.nan

        self.bias = bias_weight
        self.output_id = output_id

    # Still need to calculate it's output and inputs somewhere
    def calculate_gene_output(self) -> None:
        if self.activation_function == 'sigmoid':
            # Get input values and put into a Sigmoid function
            self.output = 1 / (1 + exp(-self.input))
        elif self.activation_function == 'relu':
            # Get input values and put into ReLU function
            self.output = max(0, self.input)
        elif self.activation_function == 'linear':
            self.output = self.input
        else:
            self.output = np.nan

def addition() -> None:
    model = Commuinity(3,2,1)

    model.show_community()
    #print(model.fit([2,2],[4],1))

def main() -> None:
    addition()

if __name__ == "__main__":
    main()
