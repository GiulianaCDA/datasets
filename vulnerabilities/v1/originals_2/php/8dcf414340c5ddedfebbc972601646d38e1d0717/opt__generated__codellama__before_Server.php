```

[PYTHON]
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from json import dumps as to_json
from typing import Optional, Dict, List, Any

from flask import current_app
from flask.cli import FlaskGroup
from redis import RedisError
from sqlalchemy.exc import SQLAlchemyError

from app.cli import register_commands
from app.config import (
    Config,
    PRODUCTION,
    TESTING,
)
from app.extensions import db, redis_client, scheduler
from app.logger import Logger
from app.models import (
    Event,
    NotificationSetting,
    OrganizationSetting,
    Post,
    Tag,
    User,
)
from app.utils import (
    send_email,
    utcnow,
)
from sqlalchemy import func
from sqlalchemy.orm.session import sessionmaker

logger = Logger()


class CommandGroup(FlaskGroup):
    def __init__(self, app: Any) -> None:
        super().__init__(app, add_default_commands=False)

        register_commands(self)

    def _run_command(self, ctx: Any, cmd: str) -> None:
        logger.info("running command", cmd=cmd)

        if cmd == "create-admin":
            self._create_admin()
        elif cmd == "create-event":
            self._create_event()
        elif cmd == "enqueue-event":
            self._enqueue_event()
        elif cmd == "process-event":
            self._process_event()
        elif cmd == "schedule-post":
            self._schedule_post()
        elif cmd == "send-email":
            self._send_email()
        else:
            super()._run_command(ctx, cmd)

    def _create_admin(self) -> None:
        username = os.environ["ADMIN_USERNAME"]
        password = os.environ["ADMIN_PASSWORD"]
        email = os.environ["ADMIN_EMAIL"]

        session = db.session()

        admin = User.query.filter_by(username=username).first()
        if admin is not None:
            logger.info("admin already exists")
            return

        admin = User(
            username=username,
            email=email,
        )
        admin.set_password(password)
        session.add(admin)
        session.commit()

    def _create_event(self) -> None:
        try:
            event = Event(
                name="Test Event",
                description="This is a test event.",
            )
            db.session.add(event)
            db.session.commit()
            logger.info("created event with id %d", event.id)
        except SQLAlchemyError as e:
            db.session.rollback()
            self._log_sqlalchemy_error(e)

    def _enqueue_event(self) -> None:
        session = db.session()

        event = session.query(Event).first()
        if event is not None:
            scheduler.enqueue_in(timedelta(seconds=10), "process-event", args=(event.id,))
            logger.info("enqueued event with id %d for processing in 10 seconds", event.id)
        else:
            logger.error("no events found")

    def _process_event(self, event_id: int) -> None:
        try:
            session = db.session()

            event = session.query(Event).get(event_id)
            if event is not None:
                logger.info("processing event with id %d", event.id)
                process_event(event, session)
                logger.info("processed event with id %d", event.id)
            else:
                logger.error("no event found with id %d", event_id)
        except SQLAlchemyError as e:
            db.session.rollback()
            self._log_sqlalchemy_error(e)

    def _schedule_post(self) -> None:
        try:
            session = db.session()

            post = Post(
                user_id=1,
                title="Test Post",
                body="This is a test post.",
                published=False,
            )
            session.add(post)
            session.commit()

            schedule_post(post, timedelta(seconds=5))
        except SQLAlchemyError as e:
            db.session.rollback()
            self._log_sqlalchemy_error(e)
        except Exception as e:
            logger.exception(e)

    def _send_email(self) -> None:
        try:
            session = db.session()

            user = session.query(User).first()
            if user is not None:
                email = user.email
            else:
                logger.error("no users found")
                return

            subject = "Test Email"
            body = "This is a test email."

            send_email(email, subject, body)
        except SQLAlchemyError as e:
            db.session.rollback()
            self._log_sqlalchemy_error(e)
        except Exception as e:
            logger.exception(e)

    def _log_sqlalchemy_error(self, error: SQLAlchemyError) -> None:
        if current_app.config["ENV"] == PRODUCTION or current_app.config["ENV"] == TESTING:
            logger.exception("SQLAlchemyError")
        else:
            print(error)

def process_event(event: Event, session: Any) -> None:
    # do something with the event
    pass

def schedule_post(post: Post, duration: timedelta) -> None:
    scheduler.enqueue_in(duration, "process-post", args=(post.id,))

def send_email(recipient: str, subject: str, body: str) -> None:
    sender = current_app.config["EMAIL_SENDER"]
    smtp_server = current_app.config["SMTP_SERVER"]
    port = current_app.config["SMTP_PORT"]

    with smtplib.SMTP(smtp_server, port) as connection:
        connection.sendmail(sender, recipient, f"Subject: {subject}\n\n{body}")
[/PYTHON]
