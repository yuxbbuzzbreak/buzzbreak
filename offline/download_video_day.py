import pandas as pd
import numpy as np
import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage

credentials, your_project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
bqclient = bigquery.Client(credentials=credentials, project=your_project_id,)
bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

query = """
select src.uid,src.video_id, src.date,DATE_DIFF(Date(src.date), Date('2020-01-01'), DAY) as diff_date, src.label,videos.category ,inaccounts.country_code
,DATE_DIFF(current_date(), Date(ubirthday.birthday), YEAR) as age,praccount.gender_input 
from (
select show.account_id as uid , show.video_id as video_id ,show.created_at as date,case when click.label is null then 0 else label end as label from
(
select account_id, safe_cast(json_extract_scalar(data, '$.id') as numeric) as video_id ,created_at

from stream_events.notification_received
where created_at > '%s'
and created_at < '%s'
and json_extract_scalar(data, '$.type') = 'video'
) as show left join 
(
select account_id, safe_cast(json_extract_scalar(data, '$.id') as numeric) as video_id, created_at,1 as label
from stream_events.notification_click
where created_at > '%s'
and created_at < '%s'
and json_extract_scalar(data, '$.type') = 'video'
) as click
on show.account_id = click.account_id
and show.video_id = click.video_id
) as src 
left join partiko.videos as videos on src.video_id = videos.id
left join input.accounts as inaccounts on src.uid = inaccounts.id
left join partiko.account_profiles_birthday as ubirthday on src.uid = ubirthday.account_id and ubirthday.birthday is not null
left join partiko.account_profiles as praccount on src.uid = praccount.account_id
limit 100
""" % ('2021-06-27','2021-06-28','2021-06-27','2021-06-28')

df = bqclient.query(query).result().to_dataframe(bqstorage_client=bqstorageclient)
print("start ")
print(df.head())
print("end ")


