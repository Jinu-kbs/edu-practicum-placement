# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 기본 설정 & 유틸
# =========================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SCHOOLS_FILE = DATA_DIR / "schools.csv"
SUBJECTS_FILE = DATA_DIR / "subjects.csv"
CAPACITY_FILE = DATA_DIR / "capacities.csv"
STUDENTS_FILE = DATA_DIR / "students.csv"
APPLICATIONS_FILE = DATA_DIR / "applications.csv"
PLACEMENTS_FILE = DATA_DIR / "placements.csv"

DEFAULT_YEAR = 2026

# -------------------------
# 초기 더미 데이터 생성
# -------------------------
def init_schools():
    if not SCHOOLS_FILE.exists():
        df = pd.DataFrame([
            {"school_id": 1, "school_name": "경상국립대 부설중학교", "school_type": "중학교"},
            {"school_id": 2, "school_name": "경상국립대 부설고등학교", "school_type": "고등학교"},
        ])
        df.to_csv(SCHOOLS_FILE, index=False)

def init_subjects():
    if not SUBJECTS_FILE.exists():
        df = pd.DataFrame([
            {"subject_id": 1, "subject_name": "국어"},
            {"subject_id": 2, "subject_name": "수학"},
            {"subject_id": 3, "subject_name": "공통과학"},
            {"subject_id": 4, "subject_name": "물리"},
            {"subject_id": 5, "subject_name": "화학"},
            {"subject_id": 6, "subject_name": "생물"},
            {"subject_id": 7, "subject_name": "지구과학"},
        ])
        df.to_csv(SUBJECTS_FILE, index=False)

def load_csv(path, columns=None):
    if path.exists():
        df = pd.read_csv(path)
        # students.csv에 birthdate 없으면 추가
        if path == STUDENTS_FILE and "birthdate" not in df.columns:
            df["birthdate"] = ""
        return df
    else:
        df = pd.DataFrame(columns=columns)
        # students.csv 스키마
        if path == STUDENTS_FILE and columns is None:
            df = pd.DataFrame(columns=["student_id", "name", "department", "major_subject_id", "birthdate"])
        return df

def save_csv(df, path):
    df.to_csv(path, index=False)

# -------------------------
# 데이터 로딩
# -------------------------
def get_schools():
    init_schools()
    return pd.read_csv(SCHOOLS_FILE)

def get_subjects():
    init_subjects()
    return pd.read_csv(SUBJECTS_FILE)

def get_capacities():
    cols = ["year", "school_id", "subject_id", "capacity"]
    return load_csv(CAPACITY_FILE, cols)

def get_students():
    df = load_csv(STUDENTS_FILE)
    if "birthdate" not in df.columns:
        df["birthdate"] = ""
    return df

def get_applications():
    cols = ["year", "student_id", "priority", "school_id", "subject_id"]
    return load_csv(APPLICATIONS_FILE, cols)

def get_placements():
    cols = ["year", "student_id", "school_id", "subject_id", "from_priority"]
    return load_csv(PLACEMENTS_FILE, cols)

# =========================
# 배정 알고리즘 (랜덤)
# =========================
def run_allocation(year: int, random_state: int = 42):
    schools = get_schools()
    subjects = get_subjects()
    capacities = get_capacities()
    applications = get_applications()
    placements = get_placements()

    caps_y = capacities[capacities["year"] == year].copy()
    apps_y = applications[applications["year"] == year].copy()

    if caps_y.empty or apps_y.empty:
        return False, "해당 학년도에 수용 인원 또는 신청 데이터가 없습니다."

    placed_y = placements[placements["year"] == year].copy()
    assigned_students = set(placed_y["student_id"]) if not placed_y.empty else set()

    new_placements = []
    rng = np.random.default_rng(random_state)

    for priority in [1, 2, 3]:
        apps_p = apps_y[
            (apps_y["priority"] == priority) &
            (~apps_y["student_id"].isin(assigned_students))
        ].copy()

        if apps_p.empty:
            continue

        for (school_id, subject_id), group in apps_p.groupby(["school_id", "subject_id"]):
            cap_row = caps_y[
                (caps_y["school_id"] == school_id) &
                (caps_y["subject_id"] == subject_id)
            ]
            if cap_row.empty:
                continue

            capacity = int(cap_row["capacity"].iloc[0])

            already_assigned = sum(
                (p["school_id"] == school_id) and (p["subject_id"] == subject_id)
                for p in new_placements
            ) + (
                0 if placed_y.empty else
                len(placed_y[(placed_y["school_id"] == school_id) &
                             (placed_y["subject_id"] == subject_id)])
            )

            available_slots = capacity - already_assigned
            if available_slots <= 0:
                continue

            group_shuffled = group.sample(
                frac=1.0,
                random_state=rng.integers(0, 1_000_000_000)
            )

            selected = group_shuffled.head(available_slots)

            for _, row in selected.iterrows():
                new_placements.append({
                    "year": year,
                    "student_id": row["student_id"],
                    "school_id": school_id,
                    "subject_id": subject_id,
                    "from_priority": priority
                })
                assigned_students.add(row["student_id"])

    if new_placements:
        new_df = pd.DataFrame(new_placements)
        placements_updated = pd.concat([placements, new_df], ignore_index=True)
        save_csv(placements_updated, PLACEMENTS_FILE)
        return True, f"{len(new_placements)}명의 학생이 자동 배정되었습니다."
    else:
        return False, "새로 배정할 학생이 없습니다."

# =========================
# UI: 학생 화면 (학번 + 생년월일 인증 + URL sid)
# =========================
def student_view(year: int):
    st.header("학생 교육실습 신청")

    schools = get_schools()
    subjects = get_subjects()
    students = get_students()
    apps = get_applications()

    # URL 파라미터에서 sid 읽기
    params = st.experimental_get_query_params()
    sid_from_url = params.get("sid", [""])[0]

    st.subheader("① 로그인 (학번 + 생년월일)")

    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("학번", value=sid_from_url)
    with col2:
        birthdate = st.text_input("생년월일 (YYYYMMDD)", type="password")

    if not student_id or not birthdate:
        st.info("학번과 생년월일(YYYYMMDD)을 입력하면 아래에 신청 화면이 표시됩니다.")
        return

    # 기존 학생 데이터 확인
    s_row = students[students["student_id"] == student_id]

    # birthdate 컬럼 보정
    if "birthdate" not in students.columns:
        students["birthdate"] = ""

    is_new_student = s_row.empty
    existing_name = ""
    existing_dept = ""
    existing_major_subject_id = subjects["subject_id"].iloc[0]  # default

    if is_new_student:
        st.info("처음 로그인하는 학생입니다. 아래에서 기본 정보를 입력하면 계정이 생성됩니다.")
    else:
        stored_birth = str(s_row["birthdate"].iloc[0]) if not pd.isna(s_row["birthdate"].iloc[0]) else ""
        if stored_birth and stored_birth != birthdate:
            st.error("학번 또는 생년월일이 올바르지 않습니다.")
            return
        existing_name = s_row["name"].iloc[0]
        existing_dept = s_row["department"].iloc[0]
        existing_major_subject_id = int(s_row["major_subject_id"].iloc[0])

    # 기존 신청 데이터
    existing_apps = apps[(apps["year"] == year) & (apps["student_id"] == student_id)]

    st.markdown("---")
    st.subheader("② 기본 정보 입력 / 수정")

    name = st.text_input("이름", value=existing_name)
    department = st.text_input("전공(학과)", value=existing_dept)

    subject_names = subjects["subject_name"].tolist()
    default_major_index = 0
    if existing_major_subject_id in subjects["subject_id"].values:
        default_major_index = int(
            subjects[subjects["subject_id"] == existing_major_subject_id].index[0]
        )

    major_subject_name = st.selectbox(
        "전공 교과(실습 교과)",
        subject_names,
        index=default_major_index
    )
    major_subject_id = int(
        subjects[subjects["subject_name"] == major_subject_name]["subject_id"].iloc[0]
    )

    st.markdown("---")
    st.subheader("③ 희망 학교/교과 선택 (1·2·3지망)")

    def get_default_choice(priority):
        if existing_apps is None or existing_apps.empty:
            return None, None
        row = existing_apps[existing_apps["priority"] == priority]
        if row.empty:
            return None, None
        return int(row["school_id"].iloc[0]), int(row["subject_id"].iloc[0])

    p1_school_default, p1_subject_default = get_default_choice(1)
    p2_school_default, p2_subject_default = get_default_choice(2)
    p3_school_default, p3_subject_default = get_default_choice(3)

    def priority_selector(label_prefix, default_school_id, default_subject_id, key_prefix):
        school_names = schools["school_name"].tolist()
        if default_school_id in schools["school_id"].values:
            school_default_index = int(
                schools[schools["school_id"] == default_school_id].index[0]
            )
        else:
            school_default_index = 0

        school_name = st.selectbox(
            f"{label_prefix} 지망 학교",
            school_names,
            index=school_default_index,
            key=f"{key_prefix}_school"
        )
        school_id = int(
            schools[schools["school_name"] == school_name]["school_id"].iloc[0]
        )

        if default_subject_id in subjects["subject_id"].values:
            subj_default_index = int(
                subjects[subjects["subject_id"] == default_subject_id].index[0]
            )
        else:
            subj_default_index = int(
                subjects[subjects["subject_id"] == major_subject_id].index[0]
            )

        subject_name = st.selectbox(
            f"{label_prefix} 지망 교과",
            subject_names,
            index=subj_default_index,
            key=f"{key_prefix}_subject"
        )
        subject_id = int(
            subjects[subjects["subject_name"] == subject_name]["subject_id"].iloc[0]
        )
        return school_id, subject_id

    p1_school_id, p1_subject_id = priority_selector("1", p1_school_default, p1_subject_default, "p1")
    p2_school_id, p2_subject_id = priority_selector("2", p2_school_default, p2_subject_default, "p2")
    p3_school_id, p3_subject_id = priority_selector("3", p3_school_default, p3_subject_default, "p3")

    st.markdown("---")
    if st.button("신청 내용 저장"):
        if not student_id or not name:
            st.error("학번과 이름은 필수입니다.")
            return

        # 1) 학생 기본정보 저장/업데이트
        students = get_students()
        students = students[students["student_id"] != student_id]
        new_student = pd.DataFrame([{
            "student_id": student_id,
            "name": name,
            "department": department,
            "major_subject_id": major_subject_id,
            "birthdate": birthdate
        }])
        save_csv(pd.concat([students, new_student], ignore_index=True), STUDENTS_FILE)

        # 2) 신청 정보 저장/업데이트
        apps = get_applications()
        apps = apps[~((apps["year"] == year) & (apps["student_id"] == student_id))]
        new_apps = pd.DataFrame([
            {"year": year, "student_id": student_id, "priority": 1,
             "school_id": p1_school_id, "subject_id": p1_subject_id},
            {"year": year, "student_id": student_id, "priority": 2,
             "school_id": p2_school_id, "subject_id": p2_subject_id},
            {"year": year, "student_id": student_id, "priority": 3,
             "school_id": p3_school_id, "subject_id": p3_subject_id},
        ])
        save_csv(pd.concat([apps, new_apps], ignore_index=True), APPLICATIONS_FILE)

        st.success("신청 내용이 저장되었습니다. 다시 접속하면 자동으로 불러옵니다.")

    st.markdown("---")
    st.subheader("④ 나의 신청/배정 현황 확인")

    apps = get_applications()
    placements = get_placements()

    apps_q = apps[(apps["year"] == year) & (apps["student_id"] == student_id)]
    placements_q = placements[(placements["year"] == year) & (placements["student_id"] == student_id)]

    st.write("### 신청 내역")
    if apps_q.empty:
        st.info("신청 내역이 없습니다.")
    else:
        merged_apps = apps_q.merge(get_schools(), on="school_id", how="left") \
                            .merge(get_subjects(), on="subject_id", how="left")
        st.dataframe(merged_apps[["priority", "school_name", "subject_name"]])

    st.write("### 배정 결과")
    if placements_q.empty:
        st.info("아직 배정 결과가 없습니다.")
    else:
        merged_plc = placements_q.merge(get_schools(), on="school_id", how="left") \
                                  .merge(get_subjects(), on="subject_id", how="left")
        st.dataframe(merged_plc[["school_name", "subject_name", "from_priority"]])

# =========================
# UI: 학교 담당자 화면
# =========================
def school_view(year: int):
    st.header("학교 담당자 - 교과별 수용 인원 입력")

    schools = get_schools()
    subjects = get_subjects()
    caps = get_capacities()

    school_name = st.selectbox("학교 선택", schools["school_name"].tolist())
    school_id = schools[schools["school_name"] == school_name]["school_id"].iloc[0]

    st.write(f"**{year}학년도 {school_name} 수용 인원 설정**")

    caps_y = caps[(caps["year"] == year) & (caps["school_id"] == school_id)].copy()

    df = subjects.merge(
        caps_y[["subject_id", "capacity"]],
        on="subject_id",
        how="left"
    )
    df["capacity"] = df["capacity"].fillna(0).astype(int)

    new_caps = []

    for _, row in df.iterrows():
        subject_id = row["subject_id"]
        subject_name = row["subject_name"]
        cap_value = st.number_input(
            f"{subject_name} 정원",
            min_value=0,
            max_value=50,
            value=int(row["capacity"]),
            step=1,
            key=f"cap_{subject_id}"
        )
        new_caps.append({
            "year": year,
            "school_id": school_id,
            "subject_id": subject_id,
            "capacity": int(cap_value)
        })

    if st.button("정원 저장"):
        caps = get_capacities()
        caps = caps[~((caps["year"] == year) & (caps["school_id"] == school_id))]
        caps_updated = pd.concat([caps, pd.DataFrame(new_caps)], ignore_index=True)
        save_csv(caps_updated, CAPACITY_FILE)
        st.success("수용 인원이 저장되었습니다.")

# =========================
# UI: 관리자 화면
#  - 대시보드
#  - 자동 배정
#  - 배정 결과
#  - 학생 목록 관리(편집)
# =========================
def admin_view(year: int):
    st.header("관리자 대시보드 및 배정 관리")

    tab1, tab2, tab3, tab4 = st.tabs([
        "정원·신청 현황",
        "자동 배정",
        "배정 결과",
        "학생 목록 관리"
    ])

    schools = get_schools()
    subjects = get_subjects()
    capacities = get_capacities()
    applications = get_applications()
    placements = get_placements()
    students = get_students()

    caps_y = capacities[capacities["year"] == year]
    apps_y = applications[applications["year"] == year]
    plc_y = placements[placements["year"] == year]

    # ---- 탭 1: 정원·신청 현황 ----
    with tab1:
        st.subheader("학교 × 교과 정원 및 신청 현황")

        if caps_y.empty:
            st.info("해당 학년도 정원 데이터가 없습니다. 학교 담당자가 정원을 먼저 입력해야 합니다.")
        else:
            matrix = caps_y.merge(schools, on="school_id", how="left") \
                           .merge(subjects, on="subject_id", how="left")

            apps_cnt = apps_y.groupby(["school_id", "subject_id"]).size().reset_index(name="applications_count")
            matrix = matrix.merge(apps_cnt, on=["school_id", "subject_id"], how="left")
            matrix["applications_count"] = matrix["applications_count"].fillna(0).astype(int)

            plc_cnt = plc_y.groupby(["school_id", "subject_id"]).size().reset_index(name="placements_count")
            matrix = matrix.merge(plc_cnt, on=["school_id", "subject_id"], how="left")
            matrix["placements_count"] = matrix["placements_count"].fillna(0).astype(int)

            display_cols = [
                "school_name", "subject_name", "capacity",
                "applications_count", "placements_count"
            ]
            st.dataframe(matrix[display_cols].sort_values(["school_name", "subject_name"]))

    # ---- 탭 2: 자동 배정 ----
    with tab2:
        st.subheader("자동 배정 실행")

        if st.button("1→2→3지망 순서로 자동 배정 실행", type="primary"):
            ok, msg = run_allocation(year)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)

    # ---- 탭 3: 배정 결과 ----
    with tab3:
        st.subheader("배정 결과 요약")

        plc_y = get_placements()
        plc_y = plc_y[plc_y["year"] == year]

        if plc_y.empty:
            st.info("해당 학년도 배정 결과가 없습니다.")
        else:
            merged_plc = plc_y.merge(schools, on="school_id", how="left") \
                              .merge(subjects, on="subject_id", how="left")
            st.dataframe(
                merged_plc[["student_id", "school_name", "subject_name", "from_priority"]]
                .sort_values(["school_name", "subject_name", "student_id"])
            )

            st.write("### 학교별 배정 인원")
            by_school = merged_plc.groupby("school_name").size().reset_index(name="배정 인원")
            st.dataframe(by_school)

    # ---- 탭 4: 학생 목록 관리 ----
    with tab4:
        st.subheader("학생 목록 관리 (이름/학과/생년월일/전공 수정 가능)")

        if students.empty:
            st.info("등록된 학생이 없습니다. 학생이 처음 로그인하면 자동으로 추가됩니다.")
        else:
            # 간단 필터
            keyword = st.text_input("이름/학과/학번 검색", "")
            df_show = students.copy()
            if keyword:
                kw = keyword.strip()
                df_show = df_show[
                    df_show["student_id"].astype(str).str.contains(kw)
                    | df_show["name"].astype(str).str.contains(kw)
                    | df_show["department"].astype(str).str.contains(kw)
                ]

            st.caption("※ 아래 표에서 직접 수정 후, '변경사항 저장' 버튼을 누르면 CSV에 반영됩니다.")
            edited = st.data_editor(
                df_show,
                num_rows="fixed",
                hide_index=True
            )

            if st.button("변경사항 저장"):
                # edited에는 필터된 결과만 있으므로, 원본 students와 merge해서 저장
                # student_id 기준으로 업데이트
                merged = students.set_index("student_id")
                edited_indexed = edited.set_index("student_id")

                # edited에 있는 학번만 업데이트
                merged.update(edited_indexed)
                merged_reset = merged.reset_index()

                save_csv(merged_reset, STUDENTS_FILE)
                st.success("학생 정보가 저장되었습니다.")

        st.markdown("---")
        st.write("### 학생용 링크 안내")
        st.info(
            "각 학생에게는 다음 형식의 링크를 줄 수 있습니다.\n\n"
            "`https://<배포된앱주소>/?role=학생&sid=학번`\n\n"
            "예: `...?role=학생&sid=2023001`\n\n"
            "학생은 해당 링크로 접속 후, 자신의 생년월일(YYYYMMDD)을 입력하면 로그인할 수 있습니다."
        )

# =========================
# 메인
# =========================
def main():
    st.set_page_config(page_title="교육실습 배정관리 시스템(MVP)", layout="wide")

    params = st.experimental_get_query_params()
    role_param = params.get("role", ["학생"])[0]

    role_options = ["학생", "학교 담당자", "관리자"]
    if role_param not in role_options:
        role_param = "학생"
    default_index = role_options.index(role_param)

    st.sidebar.title("교육실습 배정관리(MVP)")
    year = st.sidebar.selectbox("학년도 선택", [DEFAULT_YEAR, DEFAULT_YEAR + 1], index=0)
    role = st.sidebar.radio("역할 선택", role_options, index=default_index)

    if role == "학생":
        student_view(year)
    elif role == "학교 담당자":
        school_view(year)
    else:
        admin_view(year)

if __name__ == "__main__":
    main()
