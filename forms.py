from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import (BooleanField, EmailField, PasswordField, SelectField, StringField, SubmitField,
                     TextAreaField)
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional, Regexp, ValidationError

PASSWORD = [DataRequired(), Length(min=8, max=128),
            Regexp(r"^(?=.*[A-Za-z])(?=.*\d).+$", message="Use at least one letter and one digit.")]
MOBILE = [Optional(), Regexp(r"^\+?[0-9 \-]{7,20}$", message="Enter a valid mobile number.")]


class LoginForm(FlaskForm):
    email = EmailField("Email", validators=[DataRequired(), Email(), Length(max=150)])
    password = PasswordField("Password", validators=[DataRequired(), Length(max=128)])
    remember = BooleanField("Keep me signed in")
    submit = SubmitField("Sign in")


class RegisterForm(FlaskForm):
    name = StringField("Full name", validators=[DataRequired(), Length(min=2, max=100)])
    email = EmailField("Email", validators=[DataRequired(), Email(), Length(max=150)])
    mobile_number = StringField("Mobile number", validators=MOBILE)
    password = PasswordField("Password", validators=PASSWORD)
    confirm = PasswordField("Confirm password", validators=[DataRequired(), EqualTo("password", "Passwords must match.")])
    submit = SubmitField("Create account")


class ProfileForm(FlaskForm):
    name = StringField("Full name", validators=[DataRequired(), Length(min=2, max=100)])
    mobile_number = StringField("Mobile number", validators=MOBILE)
    submit = SubmitField("Save profile")


class ChangePasswordForm(FlaskForm):
    current = PasswordField("Current password", validators=[DataRequired()])
    password = PasswordField("New password", validators=PASSWORD)
    confirm = PasswordField("Confirm new password", validators=[DataRequired(), EqualTo("password", "Passwords must match.")])
    submit = SubmitField("Change password")

    def validate_password(self, field):
        if field.data == self.current.data:
            raise ValidationError("The new password must be different from the current one.")


class UploadForm(FlaskForm):
    dataset_name = StringField("Dataset name", validators=[DataRequired(), Length(max=150)])
    source_name = StringField("Source", validators=[DataRequired(), Length(max=150)],
                              default="IBM Transactions for Anti-Money Laundering (HI-Small)")
    file = FileField("Transaction file (.csv or .xlsx)",
                     validators=[FileRequired(), FileAllowed(["csv", "xlsx"], "Only .csv and .xlsx files are accepted.")])
    submit = SubmitField("Upload and validate")


class AlertReviewForm(FlaskForm):
    status = SelectField("Status", choices=[("New", "New"), ("Reviewed", "Reviewed"), ("Closed", "Closed")])
    review_remarks = TextAreaField("Investigation remarks", validators=[Optional(), Length(max=2000)])
    submit = SubmitField("Save review")


class ReportForm(FlaskForm):
    dataset_id = SelectField("Dataset", coerce=int)
    report_format = SelectField("Format", choices=[("PDF", "PDF"), ("CSV", "CSV")])
    submit = SubmitField("Generate report")


class ActionForm(FlaskForm):
    """Empty form: carries the CSRF token for POST buttons (analyse, delete, activate...)."""
    submit = SubmitField("Confirm")
