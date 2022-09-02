import time, sys

x = 1
while True:
    try:
        print(x)
        time.sleep(.3)
        x += 1
    except KeyboardInterrupt:
        print("Bye")
        break