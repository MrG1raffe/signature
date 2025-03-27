from .factory import from_word


class AlgebraBasis:
    __dict: dict
    __dim: int
    __trunc: int

    def __init__(self, dim: int, trunc: int):
        self.__dict = dict()
        self.__dim = dim
        self.__trunc = trunc

    def __getitem__(self, word: int):
        if word not in self.__dict:
            self.__dict[word] = from_word(word=word, trunc=self.__trunc, dim=self.__dim)
        return self.__dict[word]

    @property
    def trunc(self):
        return self.trunc

    @property
    def dim(self):
        return self.dim

