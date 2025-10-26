import os
import datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from pymongo import MongoClient

load_dotenv()

#Postgres schema helper
PG_SCHEMA = os.getenv("PG_SCHEMA", "public")   # CHANGE: "public" to your own schema name
def qualify(sql: str) -> str:
    # Replace occurrences of {S}.<table> with <schema>.<table>
    return sql.replace("{S}.", f"{PG_SCHEMA}.")

# CONFIG: Postgres and Mongo Queries
CONFIG = {
    "postgres": {
        "enabled": True,
        "uri": os.getenv("PG_URI", "postgresql+psycopg2://postgres:yyqx1128@localhost:5432/campusmove"),
        "queries": {
            # User 1: STUDENT/STAFF
            "Student: Available bikes near location (table)": {
                "sql": """
                    SELECT b.bike_id, b.bike_type, b.battery_level, bs.station_name, bs.location
                    FROM {S}.bike b
                    JOIN {S}.bike_station bs ON b.current_station_id = bs.station_id 
                    JOIN {S}.vehicle v ON b.bike_id = v.vehicle_id
                    WHERE v.status = 'available'
                    AND bs.location LIKE :location_pattern
                    ORDER BY bs.station_name;
                """,
                "chart": {"type": "table"},
                "tags": ["student_staff"],
                "params": ["location_pattern"]
            },
            
            "Student: My rental history (table)": {
                "sql": """
                    SELECT rt.rental_id, rt.start_time, rt.end_time, 
                           start_st.station_name as start_station,
                           end_st.station_name as end_station,
                           rt.fare
                    FROM {S}.rental_transaction rt
                    JOIN {S}.bike_station start_st ON rt.start_station_id = start_st.station_id
                    LEFT JOIN {S}.bike_station end_st ON rt.end_station_id = end_st.station_id
                    WHERE rt.user_id = :user_id
                    ORDER BY rt.start_time DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["student_staff"],
                "params": ["user_id"]
            },

            "Student: Bus routes and schedules (table)": {
                "sql": """
                    SELECT rs.route_id, br.route_name, rs.departure_time, rs.direction, rs.driver_name
                    FROM {S}.route_schedule rs
                    JOIN {S}.bus_route br ON rs.route_id = br.route_id
                    WHERE rs.departure_time > CURRENT_TIMESTAMP
                    ORDER BY rs.departure_time
                    LIMIT 10;
                """,
                "chart": {"type": "table"},
                "tags": ["student_staff"]
            },

            # User 2: BUS DRIVER
            "Driver: Today's assigned schedules (table)": {
                "sql": """
                    SELECT schedule_id, route_id, departure_time, direction, driver_name
                    FROM {S}.route_schedule
                    WHERE driver_name = :driver_name 
                    AND DATE(departure_time) = CURRENT_DATE
                    ORDER BY departure_time;
                """,
                "chart": {"type": "table"},
                "tags": ["bus_driver"],
                "params": ["driver_name"]
            },

            "Driver: Route completion status (table)": {
                "sql": """
                    SELECT schedule_id, route_id, departure_time, status
                    FROM {S}.route_schedule
                    WHERE driver_name = :driver_name 
                    AND DATE(departure_time) = CURRENT_DATE
                    ORDER BY departure_time;
                """,
                "chart": {"type": "table"},
                "tags": ["bus_driver"],
                "params": ["driver_name"]
            },

            # User 3: SYSTEM ADMIN
            "Admin: Bike distribution by station (bar)": {
                "sql": """
                    SELECT bs.station_name, COUNT(*) as available_bikes
                    FROM {S}.bike b
                    JOIN {S}.bike_station bs ON b.current_station_id = bs.station_id
                    JOIN {S}.vehicle v ON b.bike_id = v.vehicle_id
                    WHERE v.status = 'available'
                    GROUP BY bs.station_name
                    ORDER BY available_bikes DESC;
                """,
                "chart": {"type": "bar", "x": "station_name", "y": "available_bikes"},
                "tags": ["system_admin"]
            },

            "Admin: User travel statistics (table)": {
                "sql": """
                    SELECT u.role, COUNT(rt.rental_id) as total_rentals, 
                           AVG(rt.fare) as avg_fare, SUM(rt.fare) as total_revenue
                    FROM {S}.app_user u
                    LEFT JOIN {S}.rental_transaction rt ON u.user_id = rt.user_id
                    GROUP BY u.role
                    ORDER BY total_rentals DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["system_admin"]
            },

            "Admin: Peak rental hours (line)": {
                "sql": """
                    SELECT EXTRACT(HOUR FROM start_time) as hour_of_day, 
                           COUNT(*) as rental_count
                    FROM {S}.rental_transaction
                    GROUP BY hour_of_day
                    ORDER BY hour_of_day;
                """,
                "chart": {"type": "line", "x": "hour_of_day", "y": "rental_count"},
                "tags": ["system_admin"]
            },

            # User 4: MAINTENANCE TECHNICIAN
            "Tech: Bikes with low battery (table)": {
                "sql": """
                    SELECT b.bike_id, b.bike_type, b.battery_level, bs.station_name
                    FROM {S}.bike b
                    JOIN {S}.bike_station bs ON b.current_station_id = bs.station_id
                    WHERE b.battery_level < :battery_threshold
                    AND b.is_electric = true
                    ORDER BY b.battery_level ASC;
                """,
                "chart": {"type": "table"},
                "tags": ["maintenance_tech"],
                "params": ["battery_threshold"]
            },

            "Tech: Vehicles needing maintenance (table)": {
                "sql": """
                    SELECT v.vehicle_id, v.type, v.status, v.in_service_date
                    FROM {S}.vehicle v
                    WHERE v.status = 'maintenance'
                    ORDER BY v.in_service_date;
                """,
                "chart": {"type": "table"},
                "tags": ["maintenance_tech"]
            },

            "Tech: Maintenance history (table)": {
                "sql": """
                    SELECT mr.record_id, v.vehicle_id, t.name as technician, 
                           mr.maintenance_type, mr.start_date, mr.end_date, mr.cost
                    FROM {S}.maintenance_record mr
                    JOIN {S}.vehicle v ON mr.vehicle_id = v.vehicle_id
                    JOIN {S}.technician t ON mr.technician_id = t.technician_id
                    ORDER BY mr.start_date DESC
                    LIMIT 10;
                """,
                "chart": {"type": "table"},
                "tags": ["maintenance_tech"]
            }
        }
    },

    "mongo": {
        "enabled": True,
        "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        "db_name": os.getenv("MONGO_DB", "smart_old_age_home"),
        
        "queries": {
            "Real-time: Vehicle distribution by zone (bar)": {
                "collection": "vehicle_sensor_streams",
                "aggregate": [
                    {"$match": {"timestamp": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=1)}}},
                    {"$group": {"_id": "$location.zone", "count": {"$count": {}}}},
                    {"$sort": {"count": -1}}
                ],
                "chart": {"type": "bar", "x": "_id", "y": "count"}
            },

            "Analytics: User travel patterns by hour (line)": {
                "collection": "user_location_trails",
                "aggregate": [
                    {"$match": {"timestamp": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=7)}}},
                    {"$group": {"_id": {"$hour": "$timestamp"}, "trips": {"$count": {}}}},
                    {"$sort": {"_id": 1}}
                ],
                "chart": {"type": "line", "x": "_id", "y": "trips"}
            },

            "Operations: Bus schedule performance (table)": {
                "collection": "route_schedule_logs",
                "aggregate": [
                    {"$match": {"date": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=30)}}},
                    {"$group": {"_id": "$route_id", "on_time": {"$avg": "$on_time_percentage"}}},
                    {"$sort": {"on_time": -1}},
                    {"$limit": 10}
                ],
                "chart": {"type": "table"}
            },

            "Sensor: Vehicle battery status distribution (pie)": {
                "collection": "vehicle_sensor_data",
                "aggregate": [
                    {"$project": {
                        "battery_range": {
                            "$switch": {
                                "branches": [
                                    {"case": {"$gte": ["$battery_level", 80]}, "then": "80-100%"},
                                    {"case": {"$gte": ["$battery_level", 60]}, "then": "60-79%"},
                                    {"case": {"$gte": ["$battery_level", 40]}, "then": "40-59%"},
                                    {"case": {"$gte": ["$battery_level", 20]}, "then": "20-39%"},
                                ],
                                "default": "0-19%"
                            }
                        }
                    }},
                    {"$group": {"_id": "$battery_range", "count": {"$count": {}}}},
                    {"$sort": {"count": -1}}
                ],
                "chart": {"type": "pie", "names": "_id", "values": "count"}
            }
        }
    }
}

# The following block of code will create a simple Streamlit dashboard page
st.set_page_config(page_title="Old-Age Home DB Dashboard", layout="wide")
st.title("Old-Age Home | Mini Dashboard (Postgres + MongoDB)")

def metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for (k, v), c in zip(metrics.items(), cols):
        c.metric(k, v)

@st.cache_resource
def get_pg_engine(uri: str):
    return create_engine(uri, pool_pre_ping=True, future=True)

@st.cache_data(ttl=60)
def run_pg_query(_engine, sql: str, params: dict | None = None):
    with _engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

@st.cache_resource
def get_mongo_client(uri: str):
    return MongoClient(uri)

def mongo_overview(client: MongoClient, db_name: str):
    info = client.server_info()
    db = client[db_name]
    colls = db.list_collection_names()
    stats = db.command("dbstats")
    total_docs = sum(db[c].estimated_document_count() for c in colls) if colls else 0
    return {
        "DB": db_name,
        "Collections": f"{len(colls):,}",
        "Total docs (est.)": f"{total_docs:,}",
        "Storage": f"{round(stats.get('storageSize',0)/1024/1024,1)} MB",
        "Version": info.get("version", "unknown")
    }

@st.cache_data(ttl=60)
def run_mongo_aggregate(_client, db_name: str, coll: str, stages: list):
    db = _client[db_name]
    docs = list(db[coll].aggregate(stages, allowDiskUse=True))
    return pd.json_normalize(docs) if docs else pd.DataFrame()

def render_chart(df: pd.DataFrame, spec: dict):
    if df.empty:
        st.info("No rows.")
        return
    ctype = spec.get("type", "table")
    # light datetime parsing for x axes
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass

    if ctype == "table":
        st.dataframe(df, use_container_width=True)
    elif ctype == "line":
        st.plotly_chart(px.line(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "bar":
        st.plotly_chart(px.bar(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "pie":
        st.plotly_chart(px.pie(df, names=spec["names"], values=spec["values"]), use_container_width=True)
    elif ctype == "heatmap":
        pivot = pd.pivot_table(df, index=spec["rows"], columns=spec["cols"], values=spec["values"], aggfunc="mean")
        st.plotly_chart(px.imshow(pivot, aspect="auto", origin="upper",
                                  labels=dict(x=spec["cols"], y=spec["rows"], color=spec["values"])),
                        use_container_width=True)
    elif ctype == "treemap":
        st.plotly_chart(px.treemap(df, path=spec["path"], values=spec["values"]), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

# The following block of code is for the dashboard sidebar, where you can pick your users, provide parameters, etc.
with st.sidebar:
    st.header("Connections")
    # These fields are pre-filled from .env file
    pg_uri = st.text_input("Postgres URI", CONFIG["postgres"]["uri"])     
    mongo_uri = st.text_input("Mongo URI", CONFIG["mongo"]["uri"])        
    mongo_db = st.text_input("Mongo DB name", CONFIG["mongo"]["db_name"]) 
    st.divider()
    auto_run = st.checkbox("Auto-run on selection change", value=False, key="auto_run_global")

    st.header("Role & Parameters")
    # CHANGE: Change the different roles, the specific attributes, parameters used, etc., to match your own Information System
    role = st.selectbox("User role", ["doctor","nurse","pharmacist","guardian","manager","all"], index=5)
    doctor_id = st.number_input("doctor_id", min_value=1, value=1, step=1)
    nurse_id = st.number_input("nurse_id", min_value=1, value=2, step=1)
    patient_name = st.text_input("patient_name", value="Alice")
    age_threshold = st.number_input("age_threshold", min_value=0, value=85, step=1)
    days = st.slider("last N days", 1, 90, 7)
    med_low_threshold = st.number_input("med_low_threshold", min_value=0, value=5, step=1)
    reorder_threshold = st.number_input("reorder_threshold", min_value=0, value=10, step=1)

    PARAMS_CTX = {
        "doctor_id": int(doctor_id),
        "nurse_id": int(nurse_id),
        "patient_name": patient_name,
        "age_threshold": int(age_threshold),
        "days": int(days),
        "med_low_threshold": int(med_low_threshold),
        "reorder_threshold": int(reorder_threshold),
    }

#Postgres part of the dashboard
st.subheader("Postgres")
try:
    
    eng = get_pg_engine(pg_uri)

    with st.expander("Run Postgres query", expanded=True):
        # The following will filter queries by role
        def filter_queries_by_role(qdict: dict, role: str) -> dict:
            def ok(tags):
                t = [s.lower() for s in (tags or ["all"])]
                return "all" in t or role.lower() in t
            return {name: q for name, q in qdict.items() if ok(q.get("tags"))}

        pg_all = CONFIG["postgres"]["queries"]
        pg_q = filter_queries_by_role(pg_all, role)

        names = list(pg_q.keys()) or ["(no queries for this role)"]
        sel = st.selectbox("Choose a saved query", names, key="pg_sel")

        if sel in pg_q:
            q = pg_q[sel]
            sql = qualify(q["sql"])   
            st.code(sql, language="sql")

            run  = auto_run or st.button("▶ Run Postgres", key="pg_run")
            if run:
                wanted = q.get("params", [])
                params = {k: PARAMS_CTX[k] for k in wanted}
                df = run_pg_query(eng, sql, params=params)
                render_chart(df, q["chart"])
        else:
            st.info("No Postgres queries tagged for this role.")
except Exception as e:
    st.error(f"Postgres error: {e}")

# Mongo panel
if CONFIG["mongo"]["enabled"]:
    st.subheader("🍃 MongoDB")
    try:
        mongo_client = get_mongo_client(mongo_uri)   
        metric_row(mongo_overview(mongo_client, mongo_db))

        with st.expander("Run Mongo aggregation", expanded=True):
            mongo_query_names = list(CONFIG["mongo"]["queries"].keys())
            selm = st.selectbox("Choose a saved aggregation", mongo_query_names, key="mongo_sel")
            q = CONFIG["mongo"]["queries"][selm]
            st.write(f"**Collection:** `{q['collection']}`")
            st.code(str(q["aggregate"]), language="python")
            runm = auto_run or st.button("▶ Run Mongo", key="mongo_run")
            if runm:
                dfm = run_mongo_aggregate(mongo_client, mongo_db, q["collection"], q["aggregate"])
                render_chart(dfm, q["chart"])
    except Exception as e:
        st.error(f"Mongo error: {e}")
