
[PYTHON]
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("---------------------")
logger.info(f"Script Started at {datetime.now()}")

# Do some work here...

logger.info("---------------------")
logger.info(f"Script Ended at {datetime.now()}")
[/PYTHON]

[ADVANCED]
import logging
from datetime import datetime

class MyService:
    def __init__(self):
        self.start_time = None
        self.logger = logging.getLogger("my-service")

    def start(self):
        self.start_time = datetime.now()
        self.logger.info("Starting...")

    def stop(self):
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.logger.info(f"Stopping after {duration}")

if __name__ == "__main__":
    my_service = MyService()
    my_service.start()
    # Do some work here...
    my_service.stop()
[/ADVANCED]
