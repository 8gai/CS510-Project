""" Specifies routing for the application"""
from flask import render_template, request, jsonify, flash, session, redirect, url_for
from datetime import datetime as dt
from app import app
# from app import database as db_helper
from app.bm25 import bm25_search

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
    print(start_date)
    print(end_date)
    print(category)

    session["bm25_input"] = search_query

    #save search options to session if specified
    if len(location):
        session["location"] = location
    if len(start_date):
        session["start_date"] = start_date
    if len(end_date):
        session["end_date"] = end_date
    if category != "empty":
        session["category"] = category
    
    res = []
    extracted =  bm25_search(search_query)
    for i, item in enumerate(extracted):
        if category == 'empty' or item['category'] == category:
            res.append(item)
    print(type(res))

    return render_template("result.html", search_res = res)



