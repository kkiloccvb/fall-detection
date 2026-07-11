  # 1. 打开默认摄像头(编号0),拿到一个摄像头对象
  #    → 用 cv2.VideoCapture，自己查它怎么写

  # 2. 开一个循环,让它一直读画面(while True)

      # 3. 从摄像头读一帧:同时拿到"读到没有"和"这一帧图"
      #    → read() 返回两个值,用两个变量接住

      # 4. 如果没读到(读取失败),就 break 跳出循环

      # 5. 把这一帧显示在窗口里
      #    → imshow，第一个参数是窗口名(自己起),第二个是那帧图

      # 6. 等待键盘,如果按下的是 q 键,就 break 退出
      #    → waitKey(1) 拿到按键,和 ord('q') 比较。这一步漏了窗口会一闪而过

  # 7. 循环结束后:释放摄像头 + 关掉所有窗口
  #    → release() 和 destroyAllWindows()
  
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
   print("无法打开摄像头")
   exit()
while True:
   ret, frame = cap.read()
   if not ret:
       print("无法接收，退出...")
       break
   cv2.imshow('Camera', frame)
   key = cv2.waitKey(1)
   if key & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()