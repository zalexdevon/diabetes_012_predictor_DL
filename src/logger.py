import logging
import os
from datetime import datetime
import pytz


def configure_logger():
    logs_path = "artifacts/logs"
    os.makedirs(logs_path, exist_ok=True)
    date_format = "%d-%m-%Y-%H-%M-%S"

    # Get giờ Việt Nam
    utc_now = datetime.now(pytz.utc)
    vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    vietnam_time = utc_now.astimezone(vietnam_tz)

    # Get tên file
    log_file = f"{vietnam_time.strftime(date_format)}.log"
    log_file_path = os.path.join(logs_path, log_file)

    logging.basicConfig(
        filename=log_file_path,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s\n %(message)s",
        level=logging.INFO,
        force=True,  # Needed to reconfigure logging if already set
    )
    return log_file_path
