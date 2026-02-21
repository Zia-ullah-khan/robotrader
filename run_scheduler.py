import time
import subprocess
import sys
from datetime import timedelta
from datetime import datetime
import os
import json
from getAccountInfo import get_account_info
from LLM import llm

# Path to scheduler log file
SCHEDULER_LOG_FILE = 'scheduler_log.json'

def load_scheduler_log():
    """Load scheduler run log."""
    if os.path.exists(SCHEDULER_LOG_FILE):
        try:
            with open(SCHEDULER_LOG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_scheduler_log(log):
    """Save scheduler run log."""
    with open(SCHEDULER_LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2, default=str)

def log_scheduler_run(run_data):
    """Log a scheduler run to the log file."""
    log = load_scheduler_log()
    log.append(run_data)
    # Keep only last 100 runs
    if len(log) > 100:
        log = log[-100:]
    save_scheduler_log(log)

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
            # Trading hours: 9:00 AM - 4:00 PM
            market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Check if it's a weekend (Saturday=5, Sunday=6)
            if now.weekday() >= 5:
                days_until_monday = 7 - now.weekday()
                next_monday = now + timedelta(days=days_until_monday)
                next_open = next_monday.replace(hour=9, minute=0, second=0, microsecond=0)
                sleep_seconds = (next_open - now).total_seconds()
                print(f"⏰ Weekend - Market closed. Sleeping until Monday 9:00 AM ({sleep_seconds/3600:.1f} hours)")
                time.sleep(sleep_seconds)
                continue
            
            if now < market_open:
                sleep_seconds = (market_open - now).total_seconds()
                print(f"⏰ Market not open yet. Sleeping until 9:00 AM ({sleep_seconds:.0f} seconds)")
                time.sleep(sleep_seconds)
                continue
            elif now >= market_close:
                # Sleep until next day's market open (skip weekends)
                next_day = now + timedelta(days=1)
                # If next day is Saturday, skip to Monday
                if next_day.weekday() == 5:  # Saturday
                    next_day = now + timedelta(days=3)
                elif next_day.weekday() == 6:  # Sunday
                    next_day = now + timedelta(days=2)
                next_open = next_day.replace(hour=9, minute=0, second=0, microsecond=0)
                sleep_seconds = (next_open - now).total_seconds()
                print(f"⏰ Market closed. Sleeping until next market open at 9:00 AM ({sleep_seconds/3600:.1f} hours)")
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
            
            # Log this scheduler run
            run_log_entry = {
                "run_number": run_count,
                "timestamp": datetime.now().isoformat(),
                "return_code": result.returncode if 'result' in dir() else None,
                "success": result.returncode == 0 if 'result' in dir() else False,
                "stdout_length": len(result.stdout) if 'result' in dir() and result.stdout else 0,
                "stderr_length": len(result.stderr) if 'result' in dir() and result.stderr else 0
            }
            
            # After running main.py, send its output to the LLM for analysis
            llm_response = None
            try:
                account_info = get_account_info()
                # Prepare minimal stock data; main.py already handles detailed logic
                stock_data = {"symbol": "SUMMARY", "latest_indicators": {}}
                volatility = None
                prompt = f"Analyze this trading bot run output and provide a brief summary:\n\n{result.stdout if result.stdout else 'No output'}"
                llm_response = llm(account_info, stock_data, prompt, volatility)
                print("\nLLM Response:")
                print(llm_response)
                run_log_entry["llm_analysis"] = str(llm_response)[:500]  # Truncate for storage
            except Exception as e:
                print(f"❌ Error sending output to LLM: {e}")
                run_log_entry["llm_error"] = str(e)
            
            # Save the run log
            log_scheduler_run(run_log_entry)
            
            # Calculate next run time (10 min interval)
            next_run = now + timedelta(minutes=10)
            # If next run is after market close, sleep until next market open
            if next_run >= market_close:
                # Calculate next market open (skip weekends)
                next_day = now + timedelta(days=1)
                if next_day.weekday() == 5:  # Saturday
                    next_day = now + timedelta(days=3)
                elif next_day.weekday() == 6:  # Sunday
                    next_day = now + timedelta(days=2)
                next_open = next_day.replace(hour=9, minute=0, second=0, microsecond=0)
                sleep_seconds = (next_open - now).total_seconds()
                print(f"\n⏰ Next run is after market close. Sleeping until next market open at 9:00 AM ({sleep_seconds/3600:.1f} hours)")
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
