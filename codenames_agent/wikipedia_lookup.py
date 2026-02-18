import requests
import numpy as np
from datetime import datetime, timedelta

def get_weekly_wikipedia_pageviews(page_title, language="en"):
    """
    Returns total Wikipedia pageviews for a page over the past 7 days.
    
    :param page_title: Title of the Wikipedia page (e.g., "Python_(programming_language)")
    :param language: Language edition (default: "en")
    :return: Total pageviews (int)
    """
    
    # Calculate date range (last 7 full days)
    end_date = datetime.utcnow().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=6)
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    # Format title for URL
    page_title = page_title.replace(" ", "_")
    
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{language}.wikipedia.org/all-access/all-agents/"
        f"{page_title}/daily/{start_str}/{end_str}"
    )
    
    headers = {
        "User-Agent": "MyPageViewApp/1.0 (your_email@example.com)"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    
    average_views = int(sum(item["views"] for item in data["items"])/7)
    
    return average_views

# views = get_weekly_wikipedia_pageviews("Cat")
# print(f"Average daily views in the past week: {views}")