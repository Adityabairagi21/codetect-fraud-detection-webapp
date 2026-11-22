"""
CoDetect Flask application factory.

    flask --app app init-db          create all tables (MySQL schema)
    flask --app app create-admin     first-run admin account (prompts for the password)
    flask --app app run              start the development server
"""
import getpass
import os
import re

import click
from flask import Flask, redirect, render_template, url_for
from flask_login import current_user
from flask_wtf.csrf import CSRFError

from config import Config
from extensions import csrf, db, login_manager

PASSWORD_RULE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d).{8,128}$")


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    if not app.config.get("SECRET_KEY"):
        raise RuntimeError("SECRET_KEY is not set. Copy .env.example to .env and set SECRET_KEY.")
    if not app.config.get("SQLALCHEMY_DATABASE_URI"):
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and configure MySQL.")
    for key in ("UPLOAD_FOLDER", "REPORTS_FOLDER"):
        app.config[key].mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    from database.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    from routes.admin import bp as admin_bp
    from routes.analysis import bp as analysis_bp
    from routes.auth import bp as auth_bp
    from routes.user import bp as user_bp
    for bp in (auth_bp, admin_bp, user_bp, analysis_bp):
        app.register_blueprint(bp)

    @app.route("/")
    def index():
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login"))
        return redirect(url_for("admin.dashboard" if current_user.is_admin else "user.dashboard"))

    def error_page(code, title, message):
        return render_template("errors/error.html", code=code, title=title, message=message), code

    app.register_error_handler(400, lambda e: error_page(400, "Bad request", "The request could not be processed."))
    app.register_error_handler(403, lambda e: error_page(403, "Access denied",
                                                         "You do not have permission to open this page."))
    app.register_error_handler(404, lambda e: error_page(404, "Not found", "The page or record does not exist."))
    app.register_error_handler(413, lambda e: error_page(
        413, "File too large", f"Uploads are limited to {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB."))
    app.register_error_handler(CSRFError, lambda e: error_page(
        400, "Session expired", "The form expired or was submitted from another page. Please try again."))

    @app.errorhandler(500)
    def server_error(e):
        db.session.rollback()
        return error_page(500, "Something went wrong", "An unexpected error occurred. No details are shown for security.")

    @app.context_processor
    def inject_globals():
        from services.fraud_service import model_status
        return {"model_status": model_status(app)}

    register_cli(app)
    return app


def register_cli(app):
    from database.models import User

    @app.cli.command("init-db")
    def init_db():
        """Create all tables."""
        db.create_all()
        click.echo("Database tables created.")

    @app.cli.command("create-admin")
    @click.option("--name", prompt="Admin name")
    @click.option("--email", prompt="Admin email")
    @click.option("--mobile", default="", help="Mobile number (optional)")
    def create_admin(name, email, mobile):
        """Create an admin account. The password is prompted (or read from CODETECT_ADMIN_PASSWORD for
        non-interactive setup) and is never passed on the command line."""
        email = email.strip().lower()
        if db.session.query(User).filter_by(email=email).first():
            raise click.ClickException("A user with this email already exists.")
        password = os.environ.get("CODETECT_ADMIN_PASSWORD")
        interactive = password is None
        if interactive:
            password = getpass.getpass("Password (min 8 chars, letters and digits): ")
        if not PASSWORD_RULE.match(password):
            raise click.ClickException("Password must be at least 8 characters with letters and digits.")
        if interactive and getpass.getpass("Repeat password: ") != password:
            raise click.ClickException("Passwords do not match.")
        admin = User(name=name.strip(), email=email, mobile_number=mobile or None, user_type="Admin")
        admin.set_password(password)
        db.session.add(admin)
        db.session.commit()
        click.echo(f"Admin {email} created.")


if __name__ == "__main__":
    create_app().run(debug=False)
