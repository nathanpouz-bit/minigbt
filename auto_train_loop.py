import time
from trainer import train

while True:
    print("🔁 check training...")

    train()

    print("😴 sleep 1h")
    time.sleep(3600)
