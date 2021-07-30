import ConfigParser

#
def Read_ini():
    # instantiate
    config = ConfigParser.ConfigParser()
    # parse existing file
    if(len(config.read('config.ini'))==0):
	#no exit 
	init_write_ini(config)
    # read values from a section
    _ip = config.get('server', 'ip')
    _port = config.get('server', 'port')
    _resverse = config.getboolean('resverse', 'resverse')
    #int_val = config.getint('section_a', 'int_val')
    #float_val = config.getfloat('section_a', 'pi_val')
    print("_ip",_ip,"_port",_port,"_resverse",_resverse)

def init_write_ini(config):
    if 'resverse' not in config.sections():
        config.add_section('resverse')
    config.set('resverse', 'resverse', 'False')
    # add a new section and some values
    if 'server' not in config.sections():
       config.add_section('server')
    config.set('server', 'ip', '192.168.1.57')
    config.set('server', 'port', '11000')
    # save to a file
    with open('config.ini', 'w') as configfile:
    	config.write(configfile)
    #print("init end ")
#
def write_one_ini(item,key,values):
    # instantiate
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    #
    if str(item) not in config.sections():
        config.add_section(str(item))
    config.set(str(item), str(key), str(values))
    # save to a file
    with open('config.ini', 'w') as configfile:
    	config.write(configfile)
    print("add write ini end ")
def Read_one_ini(item,key,dtype):
    # instantiate
    config = ConfigParser.ConfigParser()
    # parse existing file
    if(len(config.read('config.ini'))==0):
	#no exit 
	init_write_ini(config)
    # read values from a section
    _ip = config.get('server', 'ip')
    _port = config.get('server', 'port')
    _resverse = config.getboolean('resverse', 'resverse')
    #int_val = config.getint('section_a', 'int_val')
    #float_val = config.getfloat('section_a', 'pi_val')
    print("_ip",_ip,"_port",_port,"_resverse",_resverse)
##
#    write_ini()
#Read_ini()
