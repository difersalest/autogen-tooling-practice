import requests
import pandas as pd
from coding.constant import TEXTBOOK_LIST, EXPERTS_LIST
from typing import Optional, List
import streamlit as st

def fetch_news_json(page_idx: int, list_type: str = 'all') -> dict:
    if list_type == 'all':  
        api_url = f"https://www.taipeitimes.com/ajax_json/{page_idx}/list/"
    else:
        api_url = f"https://www.taipeitimes.com/ajax_json/{page_idx}/list/{list_type}/"

    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()

def json_to_dataframe(json_data: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(json_data, orient='columns')

def fetch_all_news(start_page: int = 1,
                   end_page: int = 1,
                   list_type: str = 'all') -> pd.DataFrame:
    """
    Retrieve and compile Taipei Times news into a single DataFrame from API.

    Args:
        start_page (int): First page index to retrieve.
        end_page (int): Last page index to retrieve (inclusive).
        list_type (str): Section of news ('front', 'taiwan', etc.).

    Returns:
        pd.DataFrame: Consolidated, sorted, and deduplicated DataFrame of news items.
    """
    frames = []
    for page in range(start_page, end_page + 1):
        try:
            json_data = fetch_news_json(page, list_type)
            df = json_to_dataframe(json_data)
            frames.append(df)
        except requests.HTTPError as e:
            print(f"Failed to fetch page {page}: {e}")

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    if 'ar_id' in all_df.columns:
        all_df.sort_values(by='ar_id', ascending=False, inplace=True)
        all_df.drop_duplicates(subset='ar_id', keep='first', inplace=True)
    all_df.reset_index(drop=True, inplace=True)

    return all_df

def search_news(
    df: pd.DataFrame,
    query: Optional[str] = None,
    search_columns: Optional[List[str]] = None,
    sections: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    news_number: Optional[int] = 5
) -> pd.DataFrame:
    """
    Search a pre-fetched news DataFrame with multiple optional filters.

    Args:
        df (pd.DataFrame): DataFrame of news articles. Required columns:
            - ar_section: section name (e.g. 'Taiwan News', 'Sports', etc.)
            - ar_pubdate: publishing date string 'YYYY-MM-DD'
            - ar_head: article title
            - ar_desc: article description
            - url: article URL
        query (str, optional): Keyword or phrase to search for in text columns.
            If None, no text filtering is applied.
        search_columns (List[str], optional): Which text columns to search.
            Defaults to ['ar_head', 'ar_desc'].
        sections (List[str], optional): List of ar_section values to include.
        date_from (str, optional): Start date (inclusive) 'YYYY-MM-DD'.
        date_to (str, optional): End date (inclusive) 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: Filtered DataFrame matching all provided criteria.

    Raises:
        ValueError: If `df` is None or empty.
        KeyError: If required columns or specified search_columns are missing.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty. Fetch news first with fetch_all_news.")

    # Ensure required columns exist
    required_cols = {'ar_section', 'ar_pubdate', 'ar_head', 'ar_desc'}
    missing_req = required_cols - set(df.columns)
    if missing_req:
        raise KeyError(f"Required columns missing from DataFrame: {missing_req}")

    # Default search columns
    if search_columns is None:
        search_columns = ['ar_head', 'ar_desc']

    # Ensure search_columns exist
    missing_search = set(search_columns) - set(df.columns)
    if missing_search:
        raise KeyError(f"Search columns not found in DataFrame: {missing_search}")

    mask = pd.Series(True, index=df.index)

    # Text query filter
    if query is not None:
        text_mask = pd.Series(False, index=df.index)
        for col in search_columns:
            text_mask |= df[col].astype(str).str.contains(query, case=False, na=False)
        mask &= text_mask

    # Section filter
    if sections is not None:
        mask &= df['ar_section'].isin(sections)

    # Date range filter
    dates = pd.to_datetime(df['ar_pubdate'], errors='coerce')
    if date_from is not None:
        start = pd.to_datetime(date_from)
        mask &= (dates >= start)

    if date_to is not None:
        end = pd.to_datetime(date_to)
        mask &= (dates <= end)

    result = df[mask].reset_index(drop=True)

    if news_number is not None:
        result = result.head(news_number)

    return result

def search_expert(name: str = None,
                  discipline: str = None,
                  interest: str = None):
    results = []
    for exp in EXPERTS_LIST["EXPERTS"]:
        if ((name and name.lower() in exp["NAME"].lower()) or
            (discipline and discipline.lower() in exp["DISCIPLINE"].lower()) or
            (interest and interest.lower() in exp["INTEREST"].lower())):
            results.append(exp)
    return results or [{"error": "No matching experts found."}]

def search_textbook(title: str = None,
                    discipline: str = None,
                    related_expert: str = None):
    results = []
    for tb in TEXTBOOK_LIST["TEXTBOOKS"]:
        if ((title and title.lower() in tb["TITLE"].lower()) or
            (discipline and discipline.lower() in tb["DISCIPLINE"].lower()) or
            (related_expert and related_expert.lower() in tb["RELATED_EXPERT"].lower())):
            results.append(tb)
    return results or [{"error": "No matching textbooks found."}]