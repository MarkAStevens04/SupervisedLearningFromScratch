class DebugObject:
    """
    A master debug object. Used to assist in debugging!
    == Attributes ==
    print_out: whether the object should print or not. False by default

    == Methods ==
    p: print only when print_out is True
    """
    def __init__(self):
        self.print_out = False
        self.print_categories = {0}

    def toggle_print(self, toggle: bool):
        """
        Toggles whether the debug object should print debug messages or not.
        :param toggle:
        :return:
        """
        self.print_out = toggle

    def add_print_categories(self, categories: set):
        """
        Adds certain categories to the allowed categories to be printed
        :param categories:
        :return:
        """
        self.print_categories = self.print_categories.union(categories)

    def p(self, message: str='', category=0):
        """
        Only prints messages if toggle_print is True.
        Will only print messages if they are in the correct category.
        Default category is 0, which will always be printed if toggle_print is True
        :param message:
        :param category:
        :return:
        """
        if self.print_out and category in self.print_categories:
            print(message)



if __name__ == '__main__':
    t1 = DebugObject()
    t1.toggle_print(True)
    t1.p('hello there!')
    t1.p()
    t1.add_print_categories({1})
    t1.p(f'should be printed', 1)
    t1.p(f'should not be printed', 2)

