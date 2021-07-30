import socket
import time
class Connserver():
    def __init__(self,ip,port):
        self._ip,self._port =ip,port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rand=0
        self.reconn=False
        print("scoket create")
	self.Conn_server()
    def Conn_server(self):
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((self._ip,self._port))
            print("conn succ")
            self.rand=0
            self.reconn=False
        except Exception as e: 
            print("Sockect Connect Error:"+str(e))
    def send_msg(self,requser=1,sed=0,tr=0,sc=0,bs=0,hl=0,fl=0,rdm=0,ssp=0,bsp=0,hld=0.0):
        _msg='{"status":0,"requset":2,"sedan":'+str(sed)+',"truck":'+str(tr)+',"scooter":'+str(sc)+',"bus":'+str(bs)+',"Hlinkcar":'+str(hl)+',"Flinkcar":'+str(fl)+',"Roadnum":'+str(rdm)+',"Sspeed":'+str(ssp)+',"Bspeed":'+str(bsp)+',"Chold":'+str(hld)+',"ipCam":"rtsp_ipcam","rand":'+str(self.rand)+'}<EOF>'
        try:
            self.s.sendall(_msg)
            self.rand=self.rand+1
            #print("send_msg=",_msg)
        except Exception as e:
            print("Sockect send Error:"+str(e))
            self.reconn=True
    def get_msg(self):
        try:
            _msg=self.s.recv(1024)
            #print("get_msg=",_msg)
            return(_msg)
        except Exception as e:
            print("Sockect get Error:"+str(e))
            self.reconn=True
            return("Error")
    def close_socket(self):
        self.s.close()
        
        
#
"""
_exit=0

_sock=Connserver('192.168.1.57',12000)
#_sock.Conn_server()

_sock.send_msg()
_sock.get_msg()

while(True):
    if(_exit!=0):break;
    if(_sock.reconn):
        print("reconn")
        _sock.Conn_server()
    _sock.send_msg(rb=5,lb=3,rc=1,lc=2)
    _sock.get_msg()
    time.sleep(0.1)

print("end get")
_sock.close_socket()
print("end socket")
"""

"""
pass

_ip,_port ='192.168.1.57',12000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("socket create")
print("conn")
s.connect((_ip, _port))
print("conn succ")
rand=0
try : #s.recv(1024)
    #Set the whole string
    s.sendall('{"status":0,"requset":2,"southCount":0,"northCount":0,"northBike":0,"southBike":0,"ipCam":"rtsp_ipcam","rand":'+str(rand)+'}<EOF>')
    #s.sendall('{"status":0,"requset":2,"yolo":[0.0,0.0,0.0,0.0],"detect":[0.0,0.0,0.0,0.0],"road":[0.0,0.0,0.0,0.0],"rand":'+str(rand)+',"sign":0}<EOF>')
    rand=rand+1
    get_data=s.recv(1024)
    print("get_data",get_data)
except socket.error:
    #Send failed
    print('Send failed')
print('send msg succ')

#_www=str(input("key"))

s.sendall('{"status":0,"requset":3,"southCount":0,"northCount":0,"northBike":0,"southBike":0,"ipCam":"rtsp_ipcam","rand":'+str(rand)+'}<EOF>')
#s.sendall('{"status":0,"requset":3,"yolo":[0.0,0.0,0.0,0.0],"detect":[0.0,0.0,0.0,0.0],"road":[0.0,0.0,0.0,0.0],"rand":'+str(rand)+',"sign":0}<EOF>')
get_data=s.recv(1024)
print("get_data",get_data)    
rand=rand+1
#_www=str(input("key"))
z=0
while(True):
    z=z+1
    
    if z==1000:
        break
    s.sendall('{"status":0,"requset":1,"southCount":0,"northCount":0,"northBike":0,"southBike":0,"ipCam":"rtsp_ipcam","rand":'+str(rand)+'}<EOF>')

    #s.sendall('{"status":0,"requset":1,"yolo":[0.0,0.0,0.0,0.0],"detect":[0.0,0.0,0.0,0.0],"road":[0.0,0.0,0.0,0.0],"rand":'+str(rand)+',"sign":0}<EOF>')
    get_data=s.recv(1024)
    print("get_data",get_data)
    rand=rand+1
    #print(z)
    time.sleep(0.1)
#
print("end get")
s.close()
print("end socket")
"""
#loss conn reconn or randsame reconn

