
def get_printer(name):
    def print_it(msg=''):
        print('%s\t - %s' %(name, msg))
    return print_it

