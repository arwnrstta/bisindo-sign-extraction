import cv2
cap = cv2.VideoCapture(0) #first camera of this computer

while True:
    ok, frame = cap.read()
    #ok -> true if succesfully read
    #frame -> numpy array with height width 3(bgr)
    if not ok: break
    frame = cv2.flip(frame, 1) 
    cv2.putText(frame, "Hello BISINDO", (20, 40), #20 40 position (x,y)
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #fontsize, color (red), line thickness
    cv2.imshow("webcam", frame) #live view
    if cv2.waitKey(1) & 0xFF == ord("q") : break #wait 1 milisecond for a key press. if press q then quit
    #& 0xFF last 8 bits of the code
cap.release(); cv2.destroyAllWindows()