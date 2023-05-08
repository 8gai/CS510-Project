""" Specifies routing for the application"""
from flask import render_template, request, jsonify, flash, session, redirect, url_for
from app import app
from app import database as db_helper

# needed to use sessions
app.secret_key = "this is a great secret key"

@app.route("/")
def homepage():
    """ returns rendered homepage """
    return render_template("search.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    search_query = request.form.get("query")
    location = request.form.get("Location")
    start_date = request.form.get("startDate")
    end_date = request.form.get("endDate")
    category = request.form.get("Category")
    print(search_query)
    print(location)
    print(type(start_date))
    print(end_date)
    print(category)

    return redirect(url_for("show_result"))


#this is student home page
@app.route("/result", methods=["GET", "POST"])
def result():
    return render_template("result.html")
