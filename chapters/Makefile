CWD=$(PWD)
ROOT=$(CWD)/../
.PHONY:	all diff convert clean
all:
	cd "$(ROOT)" && make -k &&cd "$(CWD)"
diff:
	cd "$(ROOT)" && make diff &&cd "$(CWD)"
convert:
	cd "$(ROOT)" && make convert &&cd "$(CWD)"
clean:
	cd "$(ROOT)" && make clean &&cd "$(CWD)"
