import time
import requests
import zipfile
import io
import pandas as pd
import logging
from ingestion.sources.base import BaseIngester

logger = logging.getLogger(__name__)

class GDELTIngester(BaseIngester):
    """
    Polls the GDELT v2 lastupdate.txt for new 15-minute CSV exports.
    Maps GDELT news events to Platform=2.
    """
    
    def __init__(self, producer, poll_interval=300):
        super().__init__(producer)
        self.poll_interval = poll_interval
        self.last_url = None
        self.session = requests.Session()
        
        # GDELT v2 Export Format (V2.0) Column Names (subset for our needs)
        self.columns = [
            "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate", 
            "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", 
            "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code", 
            "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
            "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode", 
            "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code", 
            "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
            "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass", 
            "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
            "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
            "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
            "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
            "DATEADDED", "SOURCEURL"
        ]

    def fetch_latest(self):
        """
        Continuously polls the GDELT last update endpoint.
        Yields parsed events.
        """
        while True:
            try:
                # 1. Fetch last update URLs
                resp = self.session.get("http://data.gdeltproject.org/gdeltv2/lastupdate.txt", timeout=10)
                resp.raise_for_status()
                
                # The file has 3 lines: export, mentions, gkg. We want the export.CSV.zip
                lines = resp.text.strip().split('\n')
                export_url = None
                for line in lines:
                    if 'export.CSV.zip' in line:
                        export_url = line.split(' ')[-1]
                        break
                
                if not export_url or export_url == self.last_url:
                    time.sleep(self.poll_interval)
                    continue
                    
                logger.info(f"Downloading new GDELT export: {export_url}")
                
                # 2. Download and unzip the CSV in memory
                zip_resp = self.session.get(export_url, timeout=30)
                zip_resp.raise_for_status()
                
                with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as f:
                        # GDELT is tab-separated and has no header
                        df = pd.read_csv(f, sep='\t', header=None, names=self.columns, low_memory=False)
                
                # 3. Process events and yield
                for _, row in df.iterrows():
                    # We use the GoldsteinScale and AvgTone as metadata for the virality model
                    metadata = {
                        "goldstein_scale": float(row["GoldsteinScale"]) if pd.notnull(row["GoldsteinScale"]) else 0.0,
                        "avg_tone": float(row["AvgTone"]) if pd.notnull(row["AvgTone"]) else 0.0,
                        "num_mentions": int(row["NumMentions"]) if pd.notnull(row["NumMentions"]) else 1,
                        "actor1": str(row["Actor1Name"]),
                        "actor2": str(row["Actor2Name"]),
                        "source_url": str(row["SOURCEURL"])
                    }
                    
                    # Convert GDELT DATEADDED (YYYYMMDDHHMMSS) to unix timestamp roughly
                    try:
                        dt = pd.to_datetime(str(row["DATEADDED"]), format='%Y%m%d%H%M%S')
                        timestamp = dt.timestamp()
                    except:
                        timestamp = time.time()
                        
                    # GDELT events don't have natural "content" like a post, so we build a summary
                    content = f"EventCode {row['EventCode']} involving {row['Actor1Name']} and {row['Actor2Name']}. URL: {row['SOURCEURL']}"

                    yield self.format_event(
                        event_id=f"gdelt_{row['GLOBALEVENTID']}",
                        platform=2, # 2 = GDELT
                        timestamp=timestamp,
                        author="GDELT",
                        content=content,
                        metadata=metadata
                    )

                self.last_url = export_url

            except Exception as e:
                logger.error(f"Error in GDELT Ingester: {e}")
                time.sleep(self.poll_interval)
