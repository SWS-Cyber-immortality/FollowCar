import io
import cv2
import logging
import socketserver
from threading import Condition
from http import server
import threading

PAGE = """\
<html>
<head>
<title>Raspberry Pi - Surveillance Camera</title>
</head>
<body>
<center><h1>Raspberry Pi - Surveillance Camera</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
</body>
</html>
"""

  
class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            with self.condition:
                self.frame = buf
                self.condition.notify_all()
output = StreamingOutput()
class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
     
        
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
   

def capture_frame(output,frame):
    try:
            ret, jpeg = cv2.imencode('.jpg', frame)
            output.write(jpeg.tobytes())
    except Exception as e:
        logging.warning('Capture thread failed: %s', str(e))
    


class Preview:
    def __init__(self):
        self.frame = None
        self.address = ('', 8000)
    #   self.server = socketserver.TCPServer(self.address, StreamingHandler)
        self.server = StreamingServer(self.address, StreamingHandler)

    # @property
    # def output(self):
    #     return self._output
    
    # @output.setter
    # def output(self, value):
    #     # Optionally, you can add validation or other logic here before setting the value
    #     self._output = value

    def preview(self,frame):
    # capture = cv2.VideoCapture(0)  # Open the camera
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = frame
        try:
            server_thread = threading.Thread(target=self.server.serve_forever)
            capture_thread = threading.Thread(target=capture_frame, args=(output, self.frame))
            server_thread.daemon = True
            capture_thread.daemon = True
            server_thread.start()
            capture_thread.start()

            # while True:
            #     pass  # Keep the program running
        except KeyboardInterrupt:
            server.shutdown()
