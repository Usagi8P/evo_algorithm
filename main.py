class Commuinity():
    """A comunity contains all individuals"""
    def __init__(self,community_size, genome_size) -> None:
        self.individuals: list[Individual] = []

    def create_community(self,community_size,genome_size):
        for i in community_size:
            self.individuals.append(Individual(genome_size))

class Individual:
    """An individual is a single actor with a unique genome made of multiple genes"""
    def __init__(self,genome_size) -> None:
        self.genome: list[Gene] = []
        self.create_genome(genome_size)

    def create_genome(self,genome_size) -> None:
        for i in genome_size:
            self.genome.append(Gene(i))

class Gene:
    """Each gene is a single node with any number of inputs an output and a bias"""
    def __init__(self, id: int) -> None:
        self.id = id
        self.output: int = 0
        self.input: int = 0
        self.bias_weight: float = 0.0
        self.activation_function: str = "sigmoid"

def main():
    pass

if __name__ == "__main__":
    main()
