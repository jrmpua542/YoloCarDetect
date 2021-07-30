/home/feifan/Ypython/d_bike.py


def checkTimeNow():

 today = int(datetime.datetime.strftime(datetime.datetime.now(),'%S'))\

 checkTime = 18
 lastTime = checkTime + 12

 if today >= checkTime and today <= lastTime :
  break
