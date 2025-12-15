import time
import csv
import re
import pandas as pd
import random
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager


INPUT_XLSX = "Companies list.xlsx"
OUTPUT_CSV = "ambitionbox_likes_dislikes_with_overall_2.csv"
FIRST_N_COMPANIES = 400
MAX_PAGES_PER_COMPANY = 20
WAIT_SEC = 10

BASE = "https://www.ambitionbox.com"
REVIEWS_BASE = f"{BASE}/reviews"

def slugify_company(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def build_url(company: str) -> str:
    return f"{REVIEWS_BASE}/{slugify_company(company)}-reviews"


def new_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-web-security")
    options.add_argument("--disable-features=IsolateOrigins,site-per-process")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
    options.add_argument("--lang=en-US,en;q=0.9")
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            window.chrome = { runtime: {} };
        """
    })
    return driver


def safe_click(driver, el):
    try:
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable(el))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        time.sleep(0.3)
        el.click()
        return True
    except (ElementClickInterceptedException, StaleElementReferenceException, TimeoutException):
        try:
            driver.execute_script("arguments[0].click();", el)
            return True
        except Exception:
            return False
    except Exception:
        return False


def scroll_incremental(driver, steps=1, pause=0.6):
    last = driver.execute_script("return document.body.scrollHeight;")
    for _ in range(steps):
        driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight*0.9));")
        time.sleep(pause)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(pause)
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.4)
    return last


def click_load_more_until_gone(driver):
    while True:
        time.sleep(0.5)
        candidates = driver.find_elements(By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'load more')]")
        visible = [b for b in candidates if b.is_displayed() and b.is_enabled()]
        if not visible:
            break
        clicked_any = False
        for b in visible:
            if safe_click(driver, b):
                clicked_any = True
                time.sleep(1.5)
        if not clicked_any:
            break


def expand_read_more(driver):
    buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Read more') or contains(., 'read more')]")
    for btn in buttons:
        if btn.is_displayed():
            safe_click(driver, btn)
            time.sleep(0.2)


def extract_reviews_on_page(driver):
    results = []

    # Find all review blocks (the blocks that contain Likes/Dislikes)
    review_blocks = driver.find_elements(
        By.XPATH,
        "//*[self::div or self::section][.//h3[normalize-space()='Likes'] or .//h4[normalize-space()='Likes']]"
    )

    print(f"    Debug: {len(review_blocks)} review blocks found")

    # Process each review block
    for idx, block in enumerate(review_blocks):

        like_text, dislike_text = "NONE", "NONE"

        # Likes
        try:
            like_el = block.find_element(
                By.XPATH,
                ".//h3[normalize-space()='Likes']/following-sibling::*[1] | .//h4[normalize-space()='Likes']/following-sibling::*[1]"
            )
            like_text = " ".join(like_el.text.split())
        except NoSuchElementException:
            pass

        # Dislikes
        try:
            dis_el = block.find_element(
                By.XPATH,
                ".//h3[normalize-space()='Dislikes']/following-sibling::*[1] | .//h4[normalize-space()='Dislikes']/following-sibling::*[1]"
            )
            dislike_text = " ".join(dis_el.text.split())
        except NoSuchElementException:
            pass

        # Extract date
        review_date = "NONE"
        try:
            # Find span with "updated on" text - matches multiple possible class combinations
            date_el = block.find_element(
                By.XPATH,
                ".//span[contains(@class, 'text-xs') and contains(@class, 'text-secondary-text')]"
            )
            date_full_text = date_el.text.strip()
            if "updated on" in date_full_text.lower():
                review_date = date_full_text.lower().split("updated on")[-1].strip()
                if review_date:
                    parts = review_date.split()
                    if len(parts) >= 3:  # Should be "11 Jul 2025" format
                        parts[1] = parts[1].capitalize()  # Capitalize month
                        review_date = " ".join(parts)
            elif date_full_text and not date_full_text.lower().startswith("updated"):
                review_date = date_full_text.strip()
                
            if "updated on" in review_date.lower():
                review_date = review_date.lower().replace("updated on", "").strip()
                
        except NoSuchElementException:
            review_date = "NONE"

        # Extract overall rating - search within this review block
        overall_rating = "NONE"
        try:
            # Look for the rating span within this review block
            rating_span = block.find_element(
                By.XPATH,
                ".//span[contains(@class, 'text-primary-text') and contains(@class, 'font-pn-700') and contains(@class, 'text-sm')]"
            )
            overall_rating = rating_span.text.strip()
        except NoSuchElementException:
            overall_rating = "NONE"

        # Store review with metadata
        results.append({
            "like_text": like_text,
            "dislike_text": dislike_text,
            "overall_rating": overall_rating,
            "review_date": review_date
        })

    deduplicated = {}
    for review in results:
        key = (review["like_text"], review["dislike_text"])
        
        if key not in deduplicated:
            deduplicated[key] = review
        else:
            # Keep the review with more complete data
            existing = deduplicated[key]
            
            # Count how many fields are not "NONE"
            existing_score = sum([
                existing["overall_rating"] != "NONE",
                existing["review_date"] != "NONE"
            ])
            new_score = sum([
                review["overall_rating"] != "NONE",
                review["review_date"] != "NONE"
            ])
            
            # Replace if new review has more complete data
            if new_score > existing_score:
                deduplicated[key] = review

    return list(deduplicated.values())


def wait_for_reviews_or_end(driver):
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.XPATH, "//h3[normalize-space()='Likes'] | //h3[normalize-space()='Dislikes']"))
        )
        return True
    except TimeoutException:
        return False


def process_company(driver, company_name: str, base_url: str, csv_file: str):
    total_collected = 0
    base_url = base_url.split('?')[0].split('#')[0]

    for page_no in range(1, MAX_PAGES_PER_COMPANY + 1):
        page_url = base_url if page_no == 1 else f"{base_url}?page={page_no}"
        print(f"  Page {page_no}: Loading {page_url}")

        try:
            driver.get(page_url)
            time.sleep(random.uniform(2, 4))
        except Exception as e:
            print(f"  Error loading page {page_no}: {e}")
            break

        if not wait_for_reviews_or_end(driver):
            break

        scroll_incremental(driver)
        click_load_more_until_gone(driver)
        expand_read_more(driver)

        page_reviews = extract_reviews_on_page(driver)

        if not page_reviews:
            print(f"  Page {page_no}: No reviews found. Stopping.")
            break

        print(f"  Page {page_no}: Extracted {len(page_reviews)} reviews")

        # --- Write to CSV immediately ---
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["company", "page", "review_index_on_page", "like_text", "dislike_text", "overall_rating", "review_date", "review_url"]
            )
            for idx, r in enumerate(page_reviews, start=1):
                writer.writerow({
                    "company": company_name,
                    "page": page_no,
                    "review_index_on_page": idx,
                    "like_text": r["like_text"],
                    "dislike_text": r["dislike_text"],
                    "overall_rating": r["overall_rating"],
                    "review_date": r["review_date"],
                    "review_url": page_url
                })

        total_collected += len(page_reviews)
        print(f"  ✓ Wrote {len(page_reviews)} reviews to CSV. Total so far: {total_collected}")
        time.sleep(random.uniform(1, 3))

    try:
        df = pd.read_csv(csv_file)

        # Remove first 3 reviews (rows where review_index_on_page is 1, 2, or 3)
        mask_not_first3 = df.groupby("company")["review_index_on_page"].transform(lambda x: ~x.isin([1, 2, 3]))
        df = df[mask_not_first3]

        #Remove rows where overall_rating == "NONE"
        df = df[df["overall_rating"].astype(str).str.upper() != "NONE"]

        # Save cleaned file (overwrite)
        df.to_csv(csv_file, index=False, encoding="utf-8")
        print(f"  ✓ Cleaned CSV for {company_name}: removed first 3 reviews and 'NONE' ratings.")
    except Exception as e:
        print(f"  ⚠️ Cleanup failed for {company_name}: {e}")

    return total_collected


def main():

    if not Path(INPUT_XLSX).exists():
        print(f"ERROR: Missing input file {INPUT_XLSX}")
        return

    df = pd.read_excel(INPUT_XLSX)
    df = df.loc[163:400]
    if "company" not in df.columns:
        print("ERROR: Input Excel must have a 'company' column.")
        return

    has_links = "Review Link" in df.columns
    companies_df = df.head(FIRST_N_COMPANIES).copy()
    companies_list = companies_df["company"].dropna().astype(str).str.strip().tolist()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["company","page","review_index_on_page","like_text","dislike_text","overall_rating","review_date","review_url"])
        writer.writeheader()

    driver = None
    total_rows = 0
    RESTART_INTERVAL = 200  # Restart browser every N companies to prevent resource issues

    try:
        for i, comp in enumerate(companies_list, start=1):
            try:
                # Restart driver periodically to prevent Chrome crashes
                if driver is None or (i - 1) % RESTART_INTERVAL == 0:
                    if driver is not None:
                        print(f"\n[Restarting browser after {RESTART_INTERVAL} companies to prevent crashes...]")
                        try:
                            driver.quit()
                        except:
                            pass
                    driver = new_driver()
                
                url = ""
                if has_links and "Review Link" in companies_df.columns:
                    link_series = companies_df[companies_df["company"] == comp]["Review Link"]
                    if not link_series.empty:
                        url = str(link_series.values[0]).strip()
                if not url or pd.isna(url):
                    url = build_url(comp)

                print(f"\nScraping ({i}/{len(companies_list)}): {comp} -> {url}")
                row_count = process_company(driver, comp, url, OUTPUT_CSV)
                total_rows += row_count
                print(f"  Done: {row_count} reviews scraped.")
            except Exception as e:
                print(f"  Error for {comp}: {e}")
                # If error occurs, try to restart driver for next company
                if driver is not None:
                    try:
                        driver.quit()
                    except:
                        pass
                    driver = None
                continue
    finally:
        # Always close driver at the end
        if driver is not None:
            try:
                driver.quit()
            except:
                pass
    
    print(f"\nAll done. Total rows written: {total_rows}")


if __name__ == "__main__":
    main()
