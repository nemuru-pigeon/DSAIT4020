
import signal
import os
from src.dashboard import Dashboard


if __name__ == '__main__':
    dashboard = Dashboard()
    dashboard.define_layout()
    dashboard.app.run_server(debug=True)

    os.kill(os.getpid(), signal.SIGTERM)
