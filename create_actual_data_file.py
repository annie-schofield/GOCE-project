import xml.etree.ElementTree as ET
import csv
import sys

#input filename
INPUT_FILE = 'GO_CONS_SST_PSO_2__20091001T235945_20091002T235944_0201.DBL'
#output filename
OUTPUT_FILE = 'goce_orbit_data.csv'

#sampling rate, at the moment 1% data points (can change later)
SAMPLING_RATE = 100 

#progress reports for sanity checks
PRINT_INTERVAL = 10000 

def process_goce_xml():
    print(f"Starting processing of {INPUT_FILE}...")
    print(f"Sampling rate: Keep 1 record for every {SAMPLING_RATE} records.")
    
    #open the CSV file for writing
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        #write Header
        #creating a single ISO timestamp
        csvwriter.writerow(['timestamp', 'X', 'Y', 'Z'])

        #initialize iterparse
        #events=('end',) triggers when a closing tag </Tag> is reached
        context = ET.iterparse(INPUT_FILE, events=('end',))
        
        #determine the iterator
        context = iter(context)
        
        #get the root element (needed to clear formatting structure later)
        _, root = next(context)

        count = 0
        saved_count = 0

        for event, elem in context:
            #we only care about the 'SP3c_Record' tag
            #use .endswith to handle potential XML namespaces automatically
            if elem.tag.endswith('SP3c_Record'):
                count += 1
                
                #SAMPLING LOGIC: only process if modulo matches
                if count % SAMPLING_RATE == 0:
                    try:
                        #extract time data
                        #Path: Time_Information -> GPS_Time -> Start -> Gregorian
                        greg = elem.find('.//Gregorian')
                        year = greg.find('Year').text
                        month = greg.find('Month').text.zfill(2) # pad with 0
                        day = greg.find('Day_of_Month').text.zfill(2)
                        hour = greg.find('Hour').text.zfill(2)
                        minute = greg.find('Minute').text.zfill(2)
                        second = float(greg.find('Second').text)
                        
                        #format seconds to handle decimals cleanly
                        sec_str = f"{second:09.6f}"
                        
                        #create ISO formatted timestamp: YYYY-MM-DDTHH:MM:SS.ssssss
                        timestamp = f"{year}-{month}-{day}T{hour}:{minute}:{sec_str}"

                        #extract position data
                        #Path: List_of_Satellite_IDs -> L15 -> Position
                        #note: We search for 'Position' recursively to be safe
                        pos = elem.find('.//Position')
                        x = pos.find('X').text
                        y = pos.find('Y').text
                        z = pos.find('Z').text

                        #write row to CSV
                        csvwriter.writerow([timestamp, x, y, z])
                        saved_count += 1

                    except AttributeError as e:
                        #this catches cases where data might be missing in a record, stops funky things
                        pass

                #clears memory
                #clear the element content to free RAM
                elem.clear()
                #also clear references from the root to this element
                root.clear()
            
            #print progress periodically for sanity
            if count % PRINT_INTERVAL == 0 and count > 0:
                print(f"Processed {count} records... (Saved {saved_count})", end='\r')

    print(f"\nDone! Processed {count} total records.")
    print(f"Saved {saved_count} data points to {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        process_goce_xml()
    except FileNotFoundError:
        print(f"Error: Could not find file '{INPUT_FILE}'. Make sure it is in the same folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")