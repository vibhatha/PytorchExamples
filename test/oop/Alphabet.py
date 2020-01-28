class Alphabet:
    def __init__(self, value, key):
        self._value = value
        self._key = key

        # getting the values

    @property
    def value(self):
        print('Getting value')
        return self._value

        # setting the values

    @value.setter
    def value(self, value):
        print('Setting value to ' + value)
        self._value = value

        # deleting the values

    @value.deleter
    def value(self):
        print('Deleting value')
        del self._value

    @property
    def key1(self):
        print("Getting Key")
        return self._key

    @key1.setter
    def key1(self, key):
        print("Setting Key to " + key)
        self._key = key

    @key1.deleter
    def key1(self):
        print("Deleting Key")
        del self._key


    # passing the value


x = Alphabet('Peter', 'Person')
print(x.value, x.key1)

x.value = 'Diesel'
x.key1 = 'Actor'

del x.value
del x.key1