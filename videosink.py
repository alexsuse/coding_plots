import subprocess

class VideoSink(object) :
	def __init__( self, size, filename="output", rate=1, byteorder="bgra" ) :
		self.size = size
		cmdstring  = ('mencoder',
			'/dev/stdin',
			'-demuxer', 'rawvideo',
			'-rawvideo', 'w=%i:h=%i'%size[::-1]+":fps=%i:format=%s"%(rate,byteorder),
			'-o', filename+'.avi',
			'-ovc', 'lavc',
			)
		self.p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=False)

	def run(self, image) :
		assert image.shape == self.size
#		image.swapaxes(0,1).tofile(self.p.stdin) # should be faster but it is indeed slower
		self.p.stdin.write(image.tostring())
	def close(self) :
		self.p.stdin.close()
		

def grayscale(x, min, max):
	return 0.0 if x < min else (255 if x > max else 255*(x-min)/(max-min))
