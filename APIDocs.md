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
