import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin

BASE_URL = "https://bcpublicservice.hua.hrsmart.com/hr/ats/JobSearch/viewAll/jobSearchPaginationExternal_page:"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def get_job_links(page_url):
    response = requests.get(page_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'jobSearchResultsGrid_table'})
    
    if not table:
        return []
        
    return [urljoin(page_url, a['href']) 
            for a in table.select('tbody tr td:nth-child(3) a[href]')]

def get_job_details_1(url):
    try:
        debug = True
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        job_data = {'url': url}
        
        # Get basic form group data
        form_groups = soup.select('#job-detail .form-group.row')
        for group in form_groups:
            label = group.find(class_='job-detail-label').get_text(strip=True)
            value = group.find(class_='job-detail-input').get_text(' ', strip=True)
            job_data[label] = value
        
        # Parse job summary HTML for fields

        debug and print('extracting summary..')
        summary_div = soup.find('div', {'id': 'job_details_ats_requisition_description'})
        if summary_div:
            current_section = None
            content = []
            
            for element in summary_div.children:
                if element.name == 'p':
                    strong = element.find('strong')
                    if strong:
                        strong_text = strong.get_text(strip=True)
                        if len(strong_text) > 3:  # Avoid short bold text
                            # First, save the previous section if exists
                            if current_section and content:
                                debug and print(f'saving section {current_section} with {len(content)} items')
                                job_data[f'summ_{current_section}'] = '\n'.join(content)
                                content = []
                            
                            # Get the field/section name without colon
                            field = strong_text.rstrip(':')
                            # Get everything after the strong tag
                            value = element.get_text(strip=True)[len(strong_text):].strip(': ')
                            
                            debug and print(f'got strong, field/value: {field} / {value}')
                            
                            # If there's a value, treat as field-value pair
                            if value:
                                job_data[f'summ_{field}'] = value
                                current_section = None
                            # If no value, treat as new section
                            else:
                                current_section = field
                                print(f'new section: {current_section}')
                    
                    # Add content to current section if exists
                    elif current_section:
                        text = element.get_text(strip=True)
                        if text:
                            debug and print(f'adding to section {current_section}: {text[:50]}...')
                            content.append(text)
                
                elif element.name == 'ul' and current_section:
                    debug and print(f'processing list items for section {current_section}')
                    for li in element.find_all('li'):
                        list_item = f"• {li.get_text(strip=True)}"
                        debug and print(f'adding list item: {list_item[:50]}...')
                        content.append(list_item)
            
            # Save final section
            if current_section and content:
                debug and print(f'saving final section {current_section} with {len(content)} items')
                job_data[f'summ_{current_section}'] = '\n'.join(content)
                
        return job_data
        
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def get_job_details(url):
    try:
        debug = True
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        
        job_data = {'url': url}
        
        # Get basic form group data
        form_groups = soup.select('#job-detail .form-group.row')
        for group in form_groups:
            label = group.find(class_='job-detail-label').get_text(strip=True)
            value = group.find(class_='job-detail-input').get_text(' ', strip=True)
            job_data[label] = value
        
        # Parse job summary HTML for fields
        debug and print('extracting summary..')
        summary_div = soup.find('div', {'id': 'job_details_ats_requisition_description'})
        if summary_div:
            current_section = None
            content = []
            buffer = []

            def flush_buffer():
                if current_section and buffer:
                    joined = ' '.join(buffer).strip()
                    if joined:
                        content.append(joined)
                    buffer.clear()

            for element in summary_div.descendants:
                if element.name == 'strong':
                    strong_text = element.get_text(strip=True).rstrip(':')
                    # Only treat as heading if text is 3 or more characters
                    if len(strong_text) >= 3:
                        # Save previous section content
                        flush_buffer()
                        if current_section and content:
                            debug and print(f'saving section {current_section}')
                            job_data[f'summ_{current_section}'] = '\n'.join(content)
                            content = []
                        
                        # Start new section
                        current_section = strong_text
                        element.string = ''
                        debug and print(f'New section: {current_section}')
                    else:
                        # Treat short strong text as regular text
                        buffer.append(strong_text)
                    
                elif current_section:
                    if isinstance(element, str):
                        text = element.strip()
                        if text:
                            buffer.append(text)
                    elif element.name in ['br', 'p', 'div']:
                        flush_buffer()
                    elif element.name == 'li':
                        flush_buffer()
                        content.append(f'• {element.get_text(strip=True)}')
                    elif element.name == 'ul':
                        flush_buffer()
            
            # Save final section
            flush_buffer()
            if current_section and content:
                debug and print(f'saving final section {current_section}')
                job_data[f'summ_{current_section}'] = '\n'.join(content)
                
        return job_data
        
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None



def main_scraper():
    page = 1
    max_page=5
    all_jobs = []
    fieldnames = set()

    # First pass: Collect all possible field names
    while True:
        print("Processing page: "+str(page))
        url = f"{BASE_URL}{page}"
        job_links = get_job_links(url)
        if not job_links:
            break
        
        linkCount = 1
        
        for link in job_links:
            # link = "https://bcpublicservice.hua.hrsmart.com/hr/ats/Posting/view/119093"
            print("Processing link: "+link)
            job = get_job_details(link)
            if job:
                all_jobs.append(job)
                fieldnames.update(job.keys())

            # if linkCount==1:
            #     break
            linkCount+=1

                
        if page == max_page:
            break

        page += 1

    # Convert to sorted list for consistent ordering
    fieldnames = sorted(fieldnames)
    
    # Second pass: Write to CSV with all fields
    if all_jobs:
        with open('bc_jobs.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_jobs)

if __name__ == '__main__':
    main_scraper()
