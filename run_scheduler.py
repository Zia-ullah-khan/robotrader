import time
import subprocess
import sys
from datetime import timedelta
from datetime import datetime
import os

def run_trading_bot():
    """
    Continuously runs the main.py trading bot every 10 minutes.
    """
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        return
    
    print("RoboTrader Scheduler Started")
    print("Running main.py every 10 minutes...")
    print(f"Working directory: {os.getcwd()}")
    print("Press Ctrl+C to stop\n")
    
    run_count = 0
    
    try:
        while True:
            now = datetime.now()
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            if now < market_open:
                sleep_seconds = (market_open - now).total_seconds()
                print(f"⏰ Market not open yet. Sleeping until 9:30 AM ({sleep_seconds:.0f} seconds)")
                time.sleep(sleep_seconds)
                continue
            elif now >= market_close:
                # Sleep until next day's market open
                next_open = (now + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
                sleep_seconds = (next_open - now).total_seconds()
                print(f"⏰ Market closed. Sleeping until next market open at 9:30 AM ({sleep_seconds:.0f} seconds)")
                time.sleep(sleep_seconds)
                continue
            run_count += 1
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{'='*60}")
            print(f"RUN #{run_count} - {current_time}")
            print(f"{'='*60}")
            try:
                print("🚀 Starting main.py execution...")
                result = subprocess.run([sys.executable, "main.py"], 
                                      capture_output=True, 
                                      text=True,
                                      encoding='utf-8',
                                      errors='replace',
                                      timeout=300,
                                      cwd=os.getcwd())
                if result.returncode == 0:
                    print("✅ Trading bot executed successfully")
                    if result.stdout and result.stdout.strip():
                        print("\nOutput:")
                        print(result.stdout)
                    else:
                        print("\nOutput: (No output captured)")
                else:
                    print(f"❌ Trading bot execution failed (exit code: {result.returncode})")
                    if result.stderr and result.stderr.strip():
                        print("\nError:")
                        print(result.stderr)
                    else:
                        print("\nError: (No error details captured)")
            except subprocess.TimeoutExpired:
                print("⏰ Trading bot execution timed out after 5 minutes")
            except FileNotFoundError:
                print("❌ Error: main.py not found in current directory")
            except Exception as e:
                print(f"❌ Unexpected error running trading bot: {e}")
            # Calculate next run time (10 min interval)
            next_run = now + timedelta(minutes=10)
            # If next run is after market close, sleep until next market open
            if next_run >= market_close:
                next_open = (now + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
                sleep_seconds = (next_open - now).total_seconds()
                print(f"\n⏰ Next run is after market close. Sleeping until next market open at 9:30 AM ({sleep_seconds:.0f} seconds)")
                print(f"{'='*60}\n")
                time.sleep(sleep_seconds)
            else:
                print(f"\n⏰ Next run in 10 minutes at {next_run.strftime('%H:%M:%S')}")
                print(f"{'='*60}\n")
                time.sleep(600)
    except KeyboardInterrupt:
        print("\n\n🛑 RoboTrader Scheduler stopped by user")
        print(f"Total runs completed: {run_count}")
        print("Goodbye!")

if __name__ == "__main__":
    run_trading_bot()
