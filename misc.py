

class adder:

    def __init__(self, a, b):
        # import pdb; pdb.set_trace()
        self.x = a
        self.y = b
        self.__z = 3


    def add(self):

        return self.x + self.y
    



class adder_square(adder):


    def add(self):

        return super().add()**2
    


if __name__ == '__main__':


    obj = adder_square(3, 4)
    new_res = obj.add()

    print(obj.x)
    # print(obj.__z)
    print(obj.__dict__)