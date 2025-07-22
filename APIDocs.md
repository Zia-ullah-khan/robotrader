
# RoboTrader API Documentation

## Description

RoboTrader is an automated trading platform that leverages AI and algorithmic strategies to analyze stock market data, make trading decisions, and execute trades via the Alpaca API. The application provides both REST and WebSocket APIs for real-time and historical account information, trade decisions, and portfolio analytics. It supports:

- Automated trading decision-making using LLMs and volatility models
- Real-time trade decision broadcasting via WebSocket
- Retrieval of account status, transaction history, and portfolio performance
- Integration with Alpaca for order execution and portfolio data

RoboTrader is designed for both programmatic and interactive use, making it suitable for algo-traders, researchers, and developers who want to automate and monitor trading workflows.

## REST Endpoints

### GET `/api/account`
Returns the full account information.

**Response Example:**
```
{
  "available_balance": "...",
  "buying_power": "...",
  "stocks": [["AAPL", "100", "21258.04"], ...],
  "is_up": true,
  "status": "ACTIVE"
}
```

---

### GET `/api/status`
Returns a summary of account status.

**Response Example:**
```
{
  "available_balance": "...",
  "buying_power": "...",
  "stocks": [["AAPL", "100", "21258.04"], ...],
  "is_up": true,
  "status": "ACTIVE"
}
```

---

### GET `/api/transactions`
Returns a list of previous transactions/orders.

**Response Example:**
```
[
  {
    "id": "...",
    "symbol": "AAPL",
    "qty": "100",
    "side": "buy",
    "type": "market",
    "time_in_force": "day",
    "status": "filled",
    "filled_at": "2025-07-21T19:00:00Z",
    "submitted_at": "2025-07-21T18:59:00Z",
    "filled_qty": "100"
  },
  ...
]
```

---

### POST `/api/trade_decision`
Request a trade decision for a stock and date range.

**Request Body:**
```
{
  "symbol": "AAPL",
  "start_date": "2020-01-01",
  "end_date": "2025-07-21"
}
```

**Response Example:**
```
{
  "symbol": "AAPL",
  "volatility": 0.1234,
  "decision": {
    "action": "buy",
    "amount": 10,
    "type": "market",
    "time_in_force": "day",
    "reason": "Predicted volatility is high."
  },
  "trade_data": {
    "symbol": "AAPL",
    "qty": 10,
    "side": "buy",
    "order_type": "market",
    "time_in_force": "day"
  },
  "success": true
}
```

---

## WebSocket Events

### Connect to: `ws://localhost:5000` (Socket.IO protocol)

#### Event: `trade_decision`
- **Client emits:** `{ "symbol": "AAPL", "start_date": "2020-01-01", "end_date": "2025-07-21" }`
- **Server responds:** `trade_decision_response` with the same structure as the REST `/api/trade_decision` response.

#### Event: `algo_trade_decision`
- **Algorithm emits:**
```
{
  "server_id": "...",
  "decisions": {
    "AAPL": { "ticker": "AAPL", "quantity": 10, "reason": "..." },
    ...
  }
}
```
- **Server broadcasts:** `algo_trade_decision_broadcast` to all listeners with the same payload.

---

## Example Listener (Python)
```python
import socketio
sio = socketio.Client()
@sio.on('algo_trade_decision_broadcast')
def on_broadcast(data):
    print('Received:', data)
sio.connect('http://localhost:5000')
sio.wait()
```

---

## Notes
- All endpoints return JSON.
- WebSocket events use Socket.IO protocol.
- For `/api/trade_decision`, all fields are required in the request body.
# RoboTrader API Documentation

This document provides details on the API endpoints for the RoboTrader application.

## Base URL

The application runs on `http://127.0.0.1:5000`.

---

## Endpoints

### 1. Main Page

- **URL:** `/`
- **Method:** `GET`
- **Description:** Renders the main trading interface.
- **Response:** HTML content for the web interface.

### 2. Start Trading Analysis

- **URL:** `/start_trading`
- **Method:** `POST`
- **Description:** Starts the trading analysis pipeline for a given set of stocks. The process runs in the background.
- **Form Data:**
  - `stocks` (string, required): Comma-separated list of stock symbols (e.g., "AAPL,GOOGL,MSFT").
  - `start_date` (string, optional): Start date for historical data in YYYY-MM-DD format. Defaults to `2020-01-01`.
  - `end_date` (string, optional): End date for historical data in YYYY-MM-DD format. Defaults to the current date.
  - `demo_mode` (string, optional): Set to "on" to run in demo mode (no real trades executed).
- **Success Response (200 OK):**
  ```json
  {
    "message": "Trading pipeline started",
    "stocks": ["AAPL", "GOOGL", "MSFT"]
  }
  ```
- **Error Response (400 Bad Request):**
  ```json
  {
    "error": "Error message describing the issue."
  }
  ```

### 3. Stop Trading Analysis

- **URL:** `/stop_trading`
- **Method:** `POST`
- **Description:** Stops the currently running trading analysis pipeline.
- **Success Response (200 OK):**
  ```json
  {
    "message": "Trading pipeline stopped"
  }
  ```

### 4. Get Status

- **URL:** `/status`
- **Method:** `GET`
- **Description:** Retrieves the real-time status of the trading analysis pipeline.
- **Response:**
  ```json
  {
    "is_running": true,
    "current_stock": "GOOGL",
    "progress": 50,
    "total_stocks": 2,
    "results": [
        {
            "symbol": "AAPL",
            "volatility": 0.123,
            "decision": {
                "action": "buy",
                "reason": "...",
                "amount": 10,
                "notion": "...",
                "type": "market",
                "time_in_force": "day"
            },
            "trade_data": {
                "symbol": "AAPL",
                "qty": 10,
                "side": "buy",
                "order_type": "market",
                "time_in_force": "day"
            },
            "order_id": "some-order-id",
            "success": true
        }
    ],
    "errors": []
  }
  ```

### 5. Get Results

- **URL:** `/results`
- **Method:** `GET`
- **Description:** Retrieves the final results and any errors after the pipeline has completed.
- **Response:**
  ```json
  {
    "results": [
      {
        "symbol": "AAPL",
        "volatility": 0.123,
        "decision": { "...": "..." },
        "trade_data": { "...": "..." },
        "order_id": "some-order-id",
        "success": true
      },
      {
        "symbol": "GOOGL",
        "volatility": null,
        "decision": null,
        "trade_data": null,
        "order_id": null,
        "success": false,
        "error": "Error message"
      }
    ],
    "errors": ["Error processing GOOGL: ..."],
    "is_complete": true
  }
  ```

---

## Workflow Diagrams

These diagrams illustrate the operational flow of the RoboTrader application. You can view these diagrams rendered as images in VS Code with a Markdown previewer that supports Mermaid, or by using an online Mermaid editor.

### 1. API Interaction Sequence Diagram

This diagram shows the detailed sequence of interactions between the user's browser, the Flask web server, and the background trading pipeline.

![API Interaction Sequence Diagram](images/API%20Interaction%20Sequence%20Diagram.png)

### 2. Backend Trading Pipeline Flowchart

This diagram outlines the detailed internal logic of the backend pipeline when it processes each stock, including data flow, decision points, and error handling.

![Backend Trading Pipeline Flowchart](images/Backend%20Trading%20Pipeline%20Flowchart.png)
