(function () {
  "use strict";

  var API_ROOT = "/api/v1";
  var state = {
    currentView: "overview",
    dataPage: 1,
    dataPageSize: 10,
    dataTotalPages: 1,
    graduatePageData: null,
    graduates: [],
    selectedGraduates: new Map(),
    model: "qwen3-vl-32b",
    currentJob: null,
    currentResults: null,
    detailJobId: "",
    detailEntries: [],
    detailPage: 1,
    detailPageSize: 10,
    detailSelectedIndex: 0,
    taskStudentFilteredIndices: [],
    activeAuditPanels: {},
    pendingDeleteJob: null,
    pendingCancelJob: null,
    previewItems: [],
    previewIndex: 0,
    summaryPage: 1,
    summaryPageSize: 10,
    summaryTotalPages: 1,
    summaryItems: [],
    parametersLoaded: false,
    pollTimer: null
  };

  var PAGE_META = {
    overview: {
      eyebrow: "核查工作台",
      title: "数据概览",
      description: "查看当前毕业生数据、佐证材料与任务规模"
    },
    data: {
      eyebrow: "基础数据",
      title: "数据预览",
      description: "筛选毕业生及其佐证材料，并选择本次核查对象"
    },
    create: {
      eyebrow: "任务管理",
      title: "创建任务",
      description: "配置核查对象、任务名称并实时跟踪处理进度"
    },
    jobs: {
      eyebrow: "任务管理",
      title: "任务列表",
      description: "检索最近任务并进入核验结果详情"
    },
    detail: {
      eyebrow: "任务管理",
      title: "任务学生",
      description: "查看当前任务数据并筛选学生核验情况"
    },
    student: {
      eyebrow: "核验结果",
      title: "核验详情",
      description: "查看单个学生的材料、内容完整性与字段一致性判定"
    },
    summary: {
      eyebrow: "结果汇总",
      title: "核验汇总",
      description: "按学生汇总全部已完成核验结果"
    },
    parameters: {
      eyebrow: "系统诊断",
      title: "参数配置",
      description: "查看当前诊断模型、材料处理和核验规则的核心运行参数"
    }
  };

  function element(id) {
    return document.getElementById(id);
  }

  function showView(view, updateHistory) {
    if (!PAGE_META[view]) view = "overview";
    var previousView = state.currentView;
    state.currentView = view;
    document.body.classList.toggle("detail-view-active", view === "detail" || view === "student");
    document.querySelectorAll(".page-view").forEach(function (page) {
      page.classList.toggle("is-active", page.dataset.page === view);
    });
    document.querySelectorAll(".side-nav .nav-item").forEach(function (item) {
      var target = item.dataset.view;
      item.classList.toggle("is-active", target === view || (view === "detail" && target === "jobs"));
    });

    var meta = PAGE_META[view];
    element("pageEyebrow").textContent = meta.eyebrow;
    element("pageTitle").textContent = meta.title;
    element("pageDescription").textContent = meta.description;
    element("appSidebar").classList.remove("is-open");

    if (updateHistory !== false && window.location.hash !== "#" + view) {
      window.history.pushState({ view: view }, "", "#" + view);
    }
    if (previousView !== view) {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
    if (view === "summary" && previousView !== view) {
      loadCheckedStudentSummary();
    }
    if (view === "parameters" && (!state.parametersLoaded || previousView !== view)) {
      loadModelParameters();
    }
  }

  function viewFromLocation() {
    var view = window.location.hash.replace(/^#/, "");
    if (view === "detail" && !state.detailJobId) return "jobs";
    if (view === "student" && (!state.detailJobId || !state.detailEntries[state.detailSelectedIndex])) {
      return state.detailJobId ? "detail" : "jobs";
    }
    return PAGE_META[view] ? view : "overview";
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatNumber(value) {
    return new Intl.NumberFormat("zh-CN").format(Number(value || 0));
  }

  function formatTime(value) {
    if (!value) return "—";
    var date = new Date(value);
    return Number.isNaN(date.getTime())
      ? String(value)
      : date.toLocaleString("zh-CN", { hour12: false });
  }

  function fileExtension(name, extension) {
    if (extension) return String(extension).toLowerCase().replace(/^\./, "");
    var parts = String(name || "").split(".");
    return parts.length > 1 ? parts.pop().toLowerCase() : "";
  }

  function statusLabel(status) {
    var labels = {
      queued: "等待处理",
      running: "正在核验",
      completed: "已完成",
      failed: "执行失败",
      cancelled: "已取消",
      pass: "通过",
      review: "存疑",
      fail: "不通过",
      suspicious: "不通过",
      unsupported: "未核查",
      unverified: "未核查",
      idle: "未创建"
    };
    return labels[status] || status || "未知";
  }

  function showToast(message, isError) {
    var toast = element("toast");
    toast.textContent = message;
    toast.className = "toast show" + (isError ? " error" : "");
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(function () {
      toast.className = "toast";
    }, 3600);
  }

  async function apiFetch(path, options) {
    var response = await fetch(path, options || {});
    var contentType = response.headers.get("content-type") || "";
    var payload = contentType.indexOf("application/json") >= 0
      ? await response.json()
      : await response.text();
    if (!response.ok) {
      var detail = payload && payload.detail ? payload.detail : payload;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return payload;
  }

  async function loadHealth() {
    try {
      var health = await apiFetch("/health");
      state.model = health.configured_model || state.model;
      element("serviceDot").className = "status-dot status-dot--online";
      element("serviceText").textContent = "服务正常";
      element("modelBadge").textContent = state.model;
    } catch (error) {
      element("serviceDot").className = "status-dot status-dot--offline";
      element("serviceText").textContent = "服务异常";
      showToast("核验服务连接失败：" + error.message, true);
    }
  }

  async function loadOverview() {
    try {
      var data = await apiFetch(API_ROOT + "/dashboard/overview");
      element("graduateCount").textContent = formatNumber(data.graduate_count);
      element("provinceCount").textContent = "江苏省";
      element("schoolCount").textContent = formatNumber(data.school_count);
      element("cohortCount").textContent = formatNumber(data.graduation_cohort_count);
      element("withEvidenceCount").textContent = formatNumber(
        data.graduates_with_evidence_count
      );
      element("withoutEvidenceCount").textContent = formatNumber(
        data.graduates_without_evidence_count
      );
      element("evidenceCount").textContent = formatNumber(data.evidence_count);
      element("jobCount").textContent = formatNumber(data.total_jobs);
    } catch (error) {
      showToast("数据概览读取失败：" + error.message, true);
    }
  }

  function parameterValue(item) {
    if (item.key === "max_pages_per_document" && Number(item.value) === 0) return "不限制";
    if (typeof item.value === "boolean") return item.value ? "已启用" : "未启用";
    if (item.value == null || item.value === "") return "—";
    return String(item.value) + (item.unit ? " " + item.unit : "");
  }

  function shortVersion(value, kind) {
    var text = String(value || "").trim();
    if (!text) return "—";
    var match = text.match(/(?:^|-)v(\d+)/i);
    if (kind === "prompt") return match ? "提示词 V" + match[1] : "当前版本";
    return match ? "策略 V" + match[1] : "当前策略";
  }

  function parameterGroupIcon(key) {
    return ({ generation: "模", scheduling: "并", document: "文", decision: "核" })[key] || "参";
  }

  async function loadModelParameters() {
    var container = element("modelParameterGroups");
    container.innerHTML = '<section class="content-card parameter-loading-card"><span class="parameter-loading-dot"></span>' +
      '<div><strong>正在读取模型参数</strong><p>从当前核验服务获取实际运行配置…</p></div></section>';
    try {
      var data = await apiFetch(API_ROOT + "/system/model-parameters");
      state.parametersLoaded = true;
      element("parameterModelName").textContent = data.model || state.model;
      element("parameterServiceStatus").textContent = data.service_status === "online" ? "服务正常" : "状态未知";
      element("parameterServiceStatus").className = data.service_status === "online" ? "is-online" : "";
      element("parameterApiVersion").textContent = data.api_version || "—";
      element("parameterPolicyVersion").textContent = shortVersion(
        data.decision_policy_version, "policy"
      );
      container.innerHTML = (data.groups || []).map(function (group) {
        var items = (group.items || []).map(function (item) {
          var displayValue = item.key === "prompt_version"
            ? shortVersion(item.value, "prompt")
            : item.key === "decision_policy_version"
              ? shortVersion(item.value, "policy")
              : parameterValue(item);
          return '<article class="parameter-item"><div><span>' + escapeHtml(item.label || item.key) +
            '</span><p>' + escapeHtml(item.description || "") + '</p></div><strong>' +
            escapeHtml(displayValue) + '</strong></article>';
        }).join("");
        return '<section class="content-card parameter-group"><header><span class="parameter-group-icon parameter-group-icon--' +
          escapeHtml(group.key || "default") + '">' + parameterGroupIcon(group.key) + '</span><div><h3>' +
          escapeHtml(group.title || "运行参数") + '</h3><p>' + escapeHtml(group.description || "") +
          '</p></div></header><div class="parameter-item-list">' + items + '</div></section>';
      }).join("");
      if (!container.innerHTML) {
        container.innerHTML = '<section class="content-card parameter-error-card"><strong>暂无可展示参数</strong><p>当前服务没有返回参数分组。</p></section>';
      }
    } catch (error) {
      state.parametersLoaded = false;
      element("parameterServiceStatus").textContent = "读取失败";
      element("parameterServiceStatus").className = "is-offline";
      container.innerHTML = '<section class="content-card parameter-error-card"><strong>参数读取失败</strong><p>' +
        escapeHtml(error.message) + '</p><button class="button button--secondary" type="button" id="parameterRetryButton">重新读取</button></section>';
      element("parameterRetryButton").addEventListener("click", loadModelParameters);
    }
  }

  function setSelectOptions(id, placeholder, values) {
    var select = element(id);
    var current = select.value;
    select.innerHTML = '<option value="">' + escapeHtml(placeholder) + "</option>" +
      (values || []).map(function (value) {
        return '<option value="' + escapeHtml(value) + '">' + escapeHtml(value) + "</option>";
      }).join("");
    if ((values || []).indexOf(current) >= 0) select.value = current;
  }

  async function loadFilterOptions() {
    try {
      var data = await apiFetch(API_ROOT + "/graduates/filter-options");
      setSelectOptions("yearFilter", "全部年份", data.graduation_years);
      setSelectOptions("schoolFilter", "全部学校", data.schools);
      setSelectOptions("collegeFilter", "全部学院", data.colleges);
      setSelectOptions("materialCategoryFilter", "全部去向", data.destinations || data.material_categories);
      var province = element("provinceFilter");
      var hint = element("provinceHint");
      province.disabled = false;
      hint.classList.add("hidden");
      hint.textContent = "";
      province.innerHTML = '<option value="">江苏省</option>';
    } catch (error) {
      showToast("筛选选项读取失败：" + error.message, true);
    }
  }

  function graduateFilters() {
    return {
      province: element("provinceFilter").value,
      graduation_year: element("yearFilter").value,
      school: element("schoolFilter").value,
      college: element("collegeFilter").value,
      name: element("nameFilter").value.trim(),
      student_id: element("studentIdFilter").value.trim(),
      destination: element("materialCategoryFilter").value,
      evidence_status: element("evidenceStatusFilter").value,
      verification_status: element("verificationStatusFilter").value
    };
  }

  function graduateQueryParams(page, pageSize, filters) {
    return new URLSearchParams({
      page: String(page),
      page_size: String(pageSize),
      ...(filters || graduateFilters())
    });
  }

  async function loadGraduates() {
    element("graduateTableBody").innerHTML =
      '<tr><td colspan="7" class="empty-cell">正在加载数据…</td></tr>';
    var params = graduateQueryParams(state.dataPage, state.dataPageSize);
    try {
      var data = await apiFetch(API_ROOT + "/graduates?" + params.toString());
      state.graduates = data.items || [];
      state.graduatePageData = data;
      state.dataTotalPages = Math.max(1, Number(data.total_pages || 1));
      renderGraduateTable(data);
    } catch (error) {
      element("graduateTableBody").innerHTML =
        '<tr><td colspan="7" class="empty-cell">数据加载失败：' +
        escapeHtml(error.message) + "</td></tr>";
    }
  }

  function materialCategoryTone(category) {
    var value = String(category || "");
    if (value.indexOf("劳动合同") >= 0) return "contract";
    if (value.indexOf("就业协议") >= 0) return "agreement";
    if (value.indexOf("基层项目") >= 0) return "grassroots";
    if (value.indexOf("其他录用") >= 0) return "other-employment";
    if (value.indexOf("科研助理") >= 0 || value.indexOf("博士后") >= 0) return "research";
    if (value.indexOf("升学") >= 0) return "further-study";
    if (value.indexOf("创业") >= 0) return "entrepreneurship";
    if (value.indexOf("自由职业") >= 0) return "freelance";
    return "default";
  }

  function renderEvidenceList(graduate) {
    var evidences = graduate.evidences || [];
    if (!evidences.length) {
      return '<span class="no-evidence">无佐证材料</span>';
    }
    var visible = evidences.slice(0, 2).map(function (item, index) {
      var destination = graduate.byqxlb || "未明确毕业去向";
      var text = escapeHtml(destination + (evidences.length > 1 ? " · " + (index + 1) : ""));
      var toneClass = " material-kind material-kind--" + materialCategoryTone(destination);
      if (!item.preview_supported) {
        return '<span class="evidence-tag' + toneClass + '" title="' + escapeHtml(item.file_name) + '">' +
          text + "</span>";
      }
      return '<button class="evidence-button preview-button' + toneClass + '" type="button" ' +
        'data-id="' + item.evidence_file_id + '" data-graduate-id="' + graduate.graduate_id + '" data-name="' +
        escapeHtml(item.file_name) + '" data-ext="' + escapeHtml(item.extension) +
        '" title="' + escapeHtml(item.file_name) + '">' + text + "</button>";
    });
    if (evidences.length > 2) {
      visible.push('<span class="evidence-tag evidence-more">+' + (evidences.length - 2) + " 份</span>");
    }
    return '<div class="evidence-list">' + visible.join("") + "</div>";
  }

  function renderGraduateTable(data) {
    var body = element("graduateTableBody");
    if (!state.graduates.length) {
      body.innerHTML = '<tr><td colspan="7" class="empty-cell">没有符合条件的毕业生</td></tr>';
    } else {
      body.innerHTML = state.graduates.map(function (graduate) {
        var selected = state.selectedGraduates.has(graduate.graduate_id);
        return '<tr class="' + (selected ? "is-selected" : "") + '">' +
          '<td class="check-column"><input class="graduate-checkbox" type="checkbox" ' +
          'data-id="' + graduate.graduate_id + '"' + (selected ? " checked" : "") + "></td>" +
          '<td><div class="primary-cell"><strong>' + escapeHtml(graduate.xm || "未命名") +
          '</strong><small>' + escapeHtml(graduate.xh || "无学号") + "</small></div></td>" +
          '<td>江苏省</td>' +
          '<td><div class="school-cell"><strong>' + escapeHtml(graduate.xxmc || "未登记学校") +
          '</strong><small>' + escapeHtml(graduate.xymc || "未登记学院") + "</small></div></td>" +
          '<td><div class="primary-cell"><strong>' + escapeHtml(graduate.zymc || "未登记专业") +
          '</strong><small>' + escapeHtml(graduate.bynf || "未登记年份") + '</small></div></td>' +
          '<td><span class="status-badge status-badge--' + escapeHtml(graduate.verification_status || "unverified") + '">' +
          statusLabel(graduate.verification_status || "unverified") + '</span></td>' +
          '<td class="evidence-cell">' + renderEvidenceList(graduate) + "</td>" +
          "</tr>";
      }).join("");
    }
    element("dataTotalLabel").textContent = "共 " + formatNumber(data.total) + " 条记录";
    element("dataPageSummary").textContent =
      "第 " + data.page + " / " + Math.max(1, data.total_pages) + " 页，共 " +
      formatNumber(data.total) + " 条";
    element("dataPreviousPage").disabled = data.page <= 1;
    element("dataNextPage").disabled = data.page >= data.total_pages;
    var selectFilteredButton = element("selectFilteredButton");
    selectFilteredButton.disabled = Number(data.total || 0) === 0;
    selectFilteredButton.textContent = Number(data.total || 0)
      ? "选择筛选后全部（" + formatNumber(data.total) + "）"
      : "选择筛选后全部";
    bindGraduateRowActions();
    updateSelectionUi();
  }

  function bindGraduateRowActions() {
    document.querySelectorAll(".graduate-checkbox").forEach(function (checkbox) {
      checkbox.addEventListener("change", function () {
        var id = Number(checkbox.dataset.id);
        var graduate = state.graduates.find(function (item) {
          return item.graduate_id === id;
        });
        if (checkbox.checked && graduate) state.selectedGraduates.set(id, graduate);
        else state.selectedGraduates.delete(id);
        var row = checkbox.closest("tr");
        if (row) row.classList.toggle("is-selected", checkbox.checked);
        updateSelectionUi();
      });
    });
    document.querySelectorAll(".preview-button").forEach(function (button) {
      button.addEventListener("click", function () {
        openGraduateEvidencePreview(
          Number(button.dataset.graduateId), Number(button.dataset.id)
        );
      });
    });
  }

  function updateSelectionUi() {
    var pageIds = state.graduates.map(function (item) { return item.graduate_id; });
    var selectedOnPage = pageIds.filter(function (id) {
      return state.selectedGraduates.has(id);
    }).length;
    var pageCheckbox = element("selectPageCheckbox");
    pageCheckbox.checked = pageIds.length > 0 && selectedOnPage === pageIds.length;
    pageCheckbox.indeterminate = selectedOnPage > 0 && selectedOnPage < pageIds.length;

    var values = Array.from(state.selectedGraduates.values());
    var withEvidence = values.filter(function (item) {
      return Number(item.evidence_count || 0) > 0;
    }).length;
    element("selectionCount").textContent = "已选择 " + values.length + " 名毕业生";
    element("selectedGraduateCount").textContent = values.length;
    element("selectedWithEvidenceCount").textContent = withEvidence;
    element("selectedWithoutEvidenceCount").textContent = values.length - withEvidence;

    var preview = values.slice(0, 8).map(function (item) {
      return '<span class="selected-chip">' + escapeHtml(item.xm || "未命名") +
        " · " + escapeHtml(item.xh || "无学号") + "</span>";
    });
    if (values.length > 8) {
      preview.push('<span class="selected-more">另有 ' + (values.length - 8) + " 名</span>");
    }
    element("selectedGraduatePreview").innerHTML = preview.length
      ? preview.join("")
      : '<span class="empty-selection">请先在数据预览中选择毕业生</span>';
    updateCreateButton();
  }

  function updateCreateButton() {
    element("createTaskButton").disabled =
      state.selectedGraduates.size === 0 || !element("taskNameInput").value.trim();
  }

  function clearSelection() {
    state.selectedGraduates.clear();
    document.querySelectorAll(".graduate-checkbox").forEach(function (checkbox) {
      checkbox.checked = false;
      var row = checkbox.closest("tr");
      if (row) row.classList.remove("is-selected");
    });
    updateSelectionUi();
  }

  async function selectAllFilteredGraduates() {
    var button = element("selectFilteredButton");
    var total = Number((state.graduatePageData || {}).total || 0);
    if (!total) {
      showToast("当前筛选条件下没有可选择的毕业生", true);
      return;
    }

    var filters = graduateFilters();
    button.disabled = true;
    button.textContent = "正在选择…";

    try {
      var data = await apiFetch(
        API_ROOT + "/graduates/selection?" + new URLSearchParams(filters).toString()
      );
      (data.items || []).forEach(function (graduate) {
        var graduateId = graduate.graduate_id;
        state.selectedGraduates.set(graduateId, graduate);
      });
      if (state.graduatePageData) renderGraduateTable(state.graduatePageData);
      showToast("已选择当前筛选结果中的 " + formatNumber(data.total) + " 名毕业生");
    } catch (error) {
      showToast("选择全部筛选结果失败：" + error.message, true);
      updateSelectionUi();
    } finally {
      button.disabled = false;
      button.textContent = "选择筛选后全部（" + formatNumber(total) + "）";
    }
  }

  async function createTask() {
    var taskName = element("taskNameInput").value.trim();
    if (!taskName) {
      showToast("请填写任务名称", true);
      element("taskNameInput").focus();
      return;
    }
    if (!state.selectedGraduates.size) {
      showToast("请至少选择一名毕业生", true);
      return;
    }

    element("createTaskButton").disabled = true;
    try {
      var payload = {
        task_name: taskName,
        graduate_ids: Array.from(state.selectedGraduates.keys()),
        evidence_file_ids: [],
        force_recheck: element("forceRecheck").checked,
        options: {
          include_low_confidence_matches: false,
          model: state.model
        }
      };
      var job = await apiFetch(API_ROOT + "/employment-evidence-check/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      state.currentJob = job;
      renderLiveTask(job);
      clearSelection();
      element("taskNameInput").value = "";
      element("forceRecheck").checked = false;
      showToast("任务“" + taskName + "”已创建");
      startPolling(job.job_id);
      await Promise.allSettled([loadOverview(), loadJobs()]);
    } catch (error) {
      showToast("任务创建失败：" + error.message, true);
    } finally {
      updateCreateButton();
    }
  }

  function renderLiveTask(job) {
    state.currentJob = job;
    element("liveTaskEmpty").classList.add("hidden");
    element("liveTaskContent").classList.remove("hidden");
    element("liveTaskTitle").textContent = job.task_name || job.job_id;
    element("liveTaskId").textContent = job.job_id;
    element("liveTaskStatus").className =
      "status-badge status-badge--" + (job.status || "idle");
    element("liveTaskStatus").textContent = statusLabel(job.status);

    var total = Number(job.total_items || 0);
    var completed = Number(job.completed_items || 0);
    var failed = Number(job.failed_items || 0);
    var cancelled = Number(job.cancelled_items || 0);
    var finished = completed + failed + cancelled;
    var percent = total ? Math.round(finished / total * 100) : 100;
    element("liveProgressPercent").textContent = percent + "%";
    element("liveProgressBar").style.width = percent + "%";
    element("liveTotal").textContent = total;
    element("liveQueued").textContent = Number(job.queued_items || 0);
    element("liveRunning").textContent = Number(job.running_items || 0);
    element("liveCompleted").textContent = completed;
    element("liveFailed").textContent = failed;
    element("liveViewDetailButton").disabled = false;
    element("liveCancelTaskButton").disabled =
      ["queued", "running"].indexOf(job.status) < 0;
  }

  function startPolling(jobId) {
    window.clearInterval(state.pollTimer);
    pollJob(jobId);
    state.pollTimer = window.setInterval(function () {
      pollJob(jobId);
    }, 3000);
  }

  async function pollJob(jobId) {
    try {
      var job = await apiFetch(
        API_ROOT + "/employment-evidence-check/jobs/" + encodeURIComponent(jobId)
      );
      renderLiveTask(job);
      await loadJobs();
      if (state.detailJobId === jobId) await loadJobDetail(jobId, false);
      if (["completed", "failed", "cancelled"].indexOf(job.status) >= 0) {
        window.clearInterval(state.pollTimer);
        state.pollTimer = null;
        await Promise.allSettled([
          loadOverview(),
          state.detailJobId === jobId && ["detail", "student"].indexOf(state.currentView) >= 0
            ? loadJobDetail(jobId, false) : Promise.resolve(),
          state.currentView === "summary" ? loadCheckedStudentSummary() : Promise.resolve()
        ]);
        showToast("任务“" + (job.task_name || job.job_id) + "”" + statusLabel(job.status));
      }
    } catch (error) {
      window.clearInterval(state.pollTimer);
      state.pollTimer = null;
      showToast("任务状态更新失败：" + error.message, true);
    }
  }

  async function loadJobs() {
    try {
      var keyword = element("jobKeywordInput").value.trim();
      var params = new URLSearchParams({ limit: "18", keyword: keyword });
      var data = await apiFetch(
        API_ROOT + "/employment-evidence-check/jobs?" + params.toString()
      );
      var jobs = data.items || [];
      element("jobList").innerHTML = jobs.length
        ? jobs.map(renderJobCard).join("")
        : '<p class="empty-line">没有符合条件的任务</p>';
      document.querySelectorAll(".job-view-button").forEach(function (button) {
        button.addEventListener("click", async function () {
          var jobId = button.dataset.jobId;
          try {
            var job = await apiFetch(
              API_ROOT + "/employment-evidence-check/jobs/" + encodeURIComponent(jobId)
            );
            renderLiveTask(job);
            await loadJobDetail(jobId, true);
            if (job.status === "queued" || job.status === "running") startPolling(jobId);
          } catch (error) {
            showToast("任务读取失败：" + error.message, true);
          }
        });
      });
      document.querySelectorAll(".job-delete-button").forEach(function (button) {
        button.addEventListener("click", function () {
          openDeleteDialog(button.dataset.jobId, button.dataset.jobName);
        });
      });
      document.querySelectorAll(".job-cancel-button").forEach(function (button) {
        button.addEventListener("click", function () {
          openCancelDialog(button.dataset.jobId, button.dataset.jobName);
        });
      });
    } catch (error) {
      element("jobList").innerHTML =
        '<p class="empty-line">任务列表读取失败：' + escapeHtml(error.message) + "</p>";
    }
  }

  function renderJobCard(job) {
    var total = Number(job.total_items || 0);
    var finished = Number(job.completed_items || 0) + Number(job.failed_items || 0) +
      Number(job.cancelled_items || 0);
    var percent = total ? Math.round(finished / total * 100) : 100;
    var active = state.detailJobId === job.job_id ? " is-active" : "";
    var canDelete = ["completed", "failed"].indexOf(job.status) >= 0;
    if (job.status === "cancelled") canDelete = true;
    var canCancel = ["queued", "running"].indexOf(job.status) >= 0;
    return '<article class="job-card' + active + '" data-job-id="' +
      escapeHtml(job.job_id) + '">' +
      '<span class="job-card-title"><strong>' + escapeHtml(job.task_name || job.job_id) +
      '</strong><code>' + escapeHtml(job.job_id) + '</code></span>' +
      '<span class="job-progress-cell"><span class="job-progress-copy"><span>' +
      finished + " / " + total + ' 项</span><strong>' + percent + '%</strong></span>' +
      '<span class="job-progress-line"><span style="width:' + percent + '%"></span></span></span>' +
      '<span class="job-created-at">' + escapeHtml(formatTime(job.created_at)) + '</span>' +
      '<span class="status-badge status-badge--' + escapeHtml(job.status) + '">' +
      statusLabel(job.status) + '</span><span class="job-row-actions">' +
      '<button class="job-view-button" type="button" data-job-id="' + escapeHtml(job.job_id) + '">查看</button>' +
      (canCancel ? '<button class="job-cancel-button" type="button" data-job-id="' + escapeHtml(job.job_id) +
        '" data-job-name="' + escapeHtml(job.task_name || job.job_id) + '">取消</button>' : '') +
      '<button class="job-delete-button" type="button" data-job-id="' + escapeHtml(job.job_id) +
      '" data-job-name="' + escapeHtml(job.task_name || job.job_id) + '"' +
      (canDelete ? '' : ' disabled title="运行中的任务不能清除"') + '>清除</button></span></article>';
  }

  function summaryQuery() {
    return new URLSearchParams({
      page: String(state.summaryPage),
      page_size: String(state.summaryPageSize),
      result_status: element("summaryStatusFilter").value,
      graduation_year: element("summaryYearFilter").value,
      material_category: element("summaryCategoryFilter").value,
      keyword: element("summaryKeywordInput").value.trim()
    });
  }

  async function loadCheckedStudentSummary() {
    element("checkedStudentTableBody").innerHTML =
      '<tr><td colspan="7" class="empty-cell">正在汇总已核验结果…</td></tr>';
    try {
      var data = await apiFetch(
        API_ROOT + "/employment-evidence-check/checked-students?" + summaryQuery().toString()
      );
      state.summaryPage = Number(data.page || 1);
      state.summaryTotalPages = Number(data.total_pages || 1);
      state.summaryItems = data.items || [];
      renderCheckedStudentSummary(data);
    } catch (error) {
      element("checkedStudentTableBody").innerHTML =
        '<tr><td colspan="7" class="empty-cell">核验汇总读取失败：' +
        escapeHtml(error.message) + '</td></tr>';
      showToast("核验汇总读取失败：" + error.message, true);
    }
  }

  function setSummaryFilterOptions(options) {
    var currentYear = element("summaryYearFilter").value;
    var currentCategory = element("summaryCategoryFilter").value;
    element("summaryYearFilter").innerHTML = '<option value="">全部年份</option>' +
      (options.graduation_years || []).map(function (year) {
        return '<option value="' + escapeHtml(year) + '">' + escapeHtml(year) + '</option>';
      }).join("");
    element("summaryCategoryFilter").innerHTML = '<option value="">全部类别</option>' +
      (options.material_categories || []).map(function (category) {
        return '<option value="' + escapeHtml(category) + '">' + escapeHtml(category) + '</option>';
      }).join("");
    element("summaryYearFilter").value = currentYear;
    element("summaryCategoryFilter").value = currentCategory;
  }

  function renderCheckedStudentSummary(data) {
    var metrics = data.metrics || {};
    element("summaryCheckedCount").textContent = formatNumber(metrics.checked_students);
    element("summaryPassCount").textContent = formatNumber(metrics.pass_students);
    element("summaryReviewCount").textContent = formatNumber(
      metrics.fail_students == null ? metrics.review_students : metrics.fail_students
    );
    element("summaryMaterialCount").textContent = formatNumber(metrics.material_count);
    element("summaryResultCount").textContent = Number(data.total || 0) === Number(metrics.checked_students || 0)
      ? "共 " + formatNumber(data.total) + " 名学生 · 同一学生仅展示最近一次完整结果"
      : "筛选到 " + formatNumber(data.total) + " 名，共 " +
        formatNumber(metrics.checked_students) + " 名已核验学生";
    setSummaryFilterOptions(data.filter_options || {});
    element("checkedStudentTableBody").innerHTML = state.summaryItems.length
      ? state.summaryItems.map(renderCheckedStudentRow).join("")
      : '<tr><td colspan="6" class="empty-cell">没有符合条件的已核验学生</td></tr>';
    renderSummaryPagination(Number(data.total || 0), state.summaryTotalPages);
  }

  function checkedStudentMaterials(item) {
    var categories = item.material_categories || [];
    var visible = categories.slice(0, 2).map(function (category) {
      return '<span class="material-kind material-kind--' + materialCategoryTone(category) + '">' +
        escapeHtml(category) + '</span>';
    }).join("");
    var remaining = categories.length > 2
      ? '<small class="summary-material-more">另 ' + (categories.length - 2) + ' 类</small>'
      : '';
    return '<div class="summary-material-cell"><strong>' +
      formatNumber(item.material_count) + ' 份</strong><div>' + visible + remaining + '</div></div>';
  }

  function renderCheckedStudentRow(item) {
    var outcome = item.status === "pass" ? "pass" : "fail";
    return '<tr class="checked-student-row"><td><div class="primary-cell"><strong>' +
      escapeHtml(item.xm || "未登记姓名") + '</strong><small>' +
      escapeHtml(item.xh || "无学号") + '</small></div></td>' +
      '<td><div class="school-cell"><strong>' + escapeHtml(item.xxmc || "未登记学校") +
      '</strong><small>' + escapeHtml(item.xymc || "未登记学院") + '</small></div></td>' +
      '<td><div class="summary-graduation"><strong>' + escapeHtml(item.bynf || "—") +
      '</strong><small>' + escapeHtml(item.zymc || "未登记专业") + '</small></div></td>' +
      '<td>' + checkedStudentMaterials(item) + '</td>' +
      '<td><div class="summary-job-cell"><strong title="' + escapeHtml(item.task_name || "") + '">' +
      escapeHtml(item.task_name || item.job_id) + '</strong><small>' +
      escapeHtml(formatTime(item.completed_at)) + '</small></div></td>' +
      '<td><div class="summary-result-cell"><span class="status-badge status-badge--' + outcome + '">' +
      statusLabel(outcome) + '</span><small title="' + escapeHtml(item.summary || "") + '">' +
      escapeHtml(item.summary || "已完成材料核验") + '</small><button class="summary-detail-button" type="button" data-job-id="' +
      escapeHtml(item.job_id) + '" data-graduate-id="' + escapeHtml(item.graduate_id) +
      '">查看详情</button></div></td></tr>';
  }

  function renderSummaryPagination(total, totalPages) {
    var pagination = element("summaryPagination");
    pagination.classList.toggle("hidden", total === 0);
    if (!total) return;
    var first = (state.summaryPage - 1) * state.summaryPageSize + 1;
    var last = Math.min(state.summaryPage * state.summaryPageSize, total);
    element("summaryPageSummary").textContent =
      "第 " + state.summaryPage + " / " + totalPages + " 页 · " + first + "–" + last +
      " / " + total + " 名";
    element("summaryFirstPage").disabled = state.summaryPage <= 1;
    element("summaryPreviousPage").disabled = state.summaryPage <= 1;
    element("summaryNextPage").disabled = state.summaryPage >= totalPages;
    element("summaryLastPage").disabled = state.summaryPage >= totalPages;

    var start = Math.max(1, state.summaryPage - 2);
    var end = Math.min(totalPages, start + 4);
    start = Math.max(1, end - 4);
    var buttons = [];
    for (var page = start; page <= end; page += 1) {
      buttons.push('<button class="page-number summary-page-number' +
        (page === state.summaryPage ? " is-active" : "") + '" type="button" data-page="' +
        page + '">' + page + '</button>');
    }
    element("summaryPageNumbers").innerHTML = buttons.join("");
  }

  function changeSummaryPage(page) {
    var target = Math.min(Math.max(1, page), state.summaryTotalPages);
    if (target === state.summaryPage) return;
    state.summaryPage = target;
    loadCheckedStudentSummary();
  }

  function resetSummaryFilters() {
    element("summaryStatusFilter").value = "all";
    element("summaryYearFilter").value = "";
    element("summaryCategoryFilter").value = "";
    element("summaryKeywordInput").value = "";
    state.summaryPage = 1;
    loadCheckedStudentSummary();
  }

  function openDeleteDialog(jobId, jobName) {
    state.pendingDeleteJob = { jobId: jobId, jobName: jobName };
    element("deleteJobName").textContent = jobName || jobId;
    element("deleteJobDialog").showModal();
  }

  function openCancelDialog(jobId, jobName) {
    state.pendingCancelJob = { jobId: jobId, jobName: jobName };
    element("cancelJobName").textContent = jobName || jobId;
    element("cancelJobDialog").showModal();
  }

  async function cancelPendingJob() {
    if (!state.pendingCancelJob) return;
    var pending = state.pendingCancelJob;
    element("confirmCancelJobButton").disabled = true;
    try {
      var value = await apiFetch(
        API_ROOT + "/employment-evidence-check/jobs/" + encodeURIComponent(pending.jobId) + "/cancel",
        { method: "POST" }
      );
      element("cancelJobDialog").close();
      state.pendingCancelJob = null;
      window.clearInterval(state.pollTimer);
      state.pollTimer = null;
      if (state.currentJob && state.currentJob.job_id === pending.jobId) {
        state.currentJob = Object.assign({}, state.currentJob, value);
        renderLiveTask(state.currentJob);
      }
      showToast("任务已取消，已完成结果仍被保留");
      await loadJobs();
    } catch (error) {
      showToast("任务取消失败：" + error.message, true);
    } finally {
      element("confirmCancelJobButton").disabled = false;
    }
  }

  async function deletePendingJob() {
    if (!state.pendingDeleteJob) return;
    var pending = state.pendingDeleteJob;
    element("confirmDeleteJobButton").disabled = true;
    try {
      await apiFetch(
        API_ROOT + "/employment-evidence-check/jobs/" + encodeURIComponent(pending.jobId),
        { method: "DELETE" }
      );
      element("deleteJobDialog").close();
      state.pendingDeleteJob = null;
      if (state.detailJobId === pending.jobId) {
        state.detailJobId = "";
        state.currentResults = null;
        state.detailEntries = [];
        setStudentDetailAvailable(false);
        if (element("studentSummaryDrawer").open) element("studentSummaryDrawer").close();
      }
      showToast("任务日志已清除，原始材料未受影响");
      await Promise.allSettled([loadJobs(), loadOverview()]);
    } catch (error) {
      showToast("任务日志清除失败：" + error.message, true);
    } finally {
      element("confirmDeleteJobButton").disabled = false;
    }
  }

  async function loadJobDetail(jobId, scrollToDetail, graduateId) {
    try {
      var values = await Promise.all([
        apiFetch(API_ROOT + "/employment-evidence-check/jobs/" + encodeURIComponent(jobId)),
        apiFetch(API_ROOT + "/employment-evidence-check/jobs/" + encodeURIComponent(jobId) + "/results")
      ]);
      var job = values[0];
      var results = values[1];
      var isNewJob = state.detailJobId !== jobId;
      state.currentJob = job;
      state.detailJobId = jobId;
      state.currentResults = results;
      state.detailEntries = buildDetailEntries(results);
      if (isNewJob) {
        state.detailPage = 1;
        state.detailSelectedIndex = 0;
        resetTaskStudentFilters(false);
        populateTaskStudentYearFilter();
      }
      if (graduateId != null) {
        var requestedIndex = state.detailEntries.findIndex(function (entry) {
          return String((entry.student || {}).graduate_id) === String(graduateId);
        });
        if (requestedIndex < 0) {
          throw new Error("当前任务中未找到该学生的核验结果");
        }
        state.detailSelectedIndex = requestedIndex;
      }
      renderJobDetail(job);
      if (state.currentView === "student" || graduateId != null) renderStudentDetailPage();
      if (graduateId != null) {
        showView("student", true);
        return;
      }
      if (scrollToDetail) {
        showView("detail", Boolean(scrollToDetail));
      }
    } catch (error) {
      showToast("任务详情读取失败：" + error.message, true);
    }
  }

  function buildDetailEntries(results) {
    var entries = (results.students || []).map(function (student) {
      return { student: student, evidences: student.evidence_results || [] };
    });
    (results.unmatched_evidence_results || []).forEach(function (evidence, index) {
      entries.push({
        student: {
          graduate_id: "unmatched-" + index,
          xh: "",
          xm: "未关联毕业生",
          xxmc: "",
          xymc: "",
          zymc: "",
          bynf: "",
          byqxlb: ""
        },
        evidences: [evidence],
        unmatched: true
      });
    });
    return entries;
  }

  function graduateOutcome(entry) {
    var evidences = entry.evidences || [];
    if (!evidences.length || evidences.every(function (item) {
      return !item.evidence_file_id;
    })) return "unverified";
    if (evidences.some(function (item) {
      return ["queued", "running", "failed"].indexOf(item.item_status) >= 0;
    })) return "unverified";
    var aggregate = (entry.student || {}).student_result || {};
    if (aggregate.is_complete !== false && aggregate.status) {
      if (aggregate.status === "pass") return "pass";
      if (["review", "fail"].indexOf(aggregate.status) >= 0) return "fail";
      return "unverified";
    }
    if (evidences.some(function (item) {
      return item.status === "review" || item.status === "fail";
    })) return "fail";
    if (evidences.every(function (item) { return item.status === "pass"; })) return "pass";
    return "unverified";
  }

  function renderJobDetail(job) {
    element("taskDetail").classList.remove("hidden");
    element("taskDetailTitle").textContent = job.task_name || job.job_id;
    element("taskDetailMeta").textContent =
      job.job_id + " · 创建于 " + formatTime(job.created_at) + " · 模型 " + (job.model || state.model);
    element("taskDetailStatus").className =
      "status-badge status-badge--" + (job.status || "idle");
    element("taskDetailStatus").textContent = statusLabel(job.status);

    var counts = { pass: 0, fail: 0, unverified: 0 };
    state.detailEntries.forEach(function (entry) {
      counts[graduateOutcome(entry)] += 1;
    });
    element("detailTotalCount").textContent = state.detailEntries.length;
    element("detailPassCount").textContent = counts.pass;
    element("detailReviewCount").textContent = counts.fail;
    element("detailPendingCount").textContent = counts.unverified;
    renderDetailPage();
    loadJobsActiveState();
  }

  function loadJobsActiveState() {
    document.querySelectorAll(".job-card").forEach(function (card) {
      card.classList.toggle("is-active", card.dataset.jobId === state.detailJobId);
    });
  }

  function renderDetailPage() {
    var records = filteredTaskStudentRecords();
    state.taskStudentFilteredIndices = records.map(function (record) { return record.index; });
    var total = records.length;
    var totalPages = Math.max(1, Math.ceil(total / state.detailPageSize));
    state.detailPage = Math.min(Math.max(1, state.detailPage), totalPages);
    var start = (state.detailPage - 1) * state.detailPageSize;
    var pageRecords = records.slice(start, start + state.detailPageSize);
    element("taskStudentResultCount").textContent = state.detailEntries.length === total
      ? "共 " + total + " 名学生 · 点击任意学生查看核验摘要"
      : "筛选到 " + total + " 名，共 " + state.detailEntries.length + " 名 · 点击学生查看摘要";
    element("taskStudentTableBody").innerHTML = pageRecords.length
      ? pageRecords.map(renderTaskStudentRow).join("")
      : '<tr><td colspan="6" class="empty-cell">没有符合筛选条件的学生</td></tr>';
    bindTaskStudentRows();
    renderDetailPagination(total, totalPages);
  }

  function populateTaskStudentYearFilter() {
    var select = element("taskStudentYearFilter");
    var years = Array.from(new Set(state.detailEntries.map(function (entry) {
      return String((entry.student || {}).bynf || "").trim();
    }).filter(Boolean))).sort().reverse();
    select.innerHTML = '<option value="">全部年份</option>' + years.map(function (year) {
      return '<option value="' + escapeHtml(year) + '">' + escapeHtml(year) + '</option>';
    }).join("");
  }

  function resetTaskStudentFilters(shouldRender) {
    ["taskStudentYearFilter", "taskStudentNameFilter", "taskStudentIdFilter", "taskStudentCollegeFilter"]
      .forEach(function (id) {
        if (element(id)) element(id).value = "";
      });
    if (element("taskStudentStatusFilter")) element("taskStudentStatusFilter").value = "all";
    state.detailPage = 1;
    if (shouldRender !== false && state.detailJobId) renderDetailPage();
  }

  function filteredTaskStudentRecords() {
    var status = element("taskStudentStatusFilter").value;
    var year = element("taskStudentYearFilter").value;
    var name = element("taskStudentNameFilter").value.trim().toLowerCase();
    var studentId = element("taskStudentIdFilter").value.trim().toLowerCase();
    var college = element("taskStudentCollegeFilter").value.trim().toLowerCase();
    return state.detailEntries.map(function (entry, index) {
      return { entry: entry, index: index };
    }).filter(function (record) {
      var student = record.entry.student || {};
      return (status === "all" || graduateOutcome(record.entry) === status) &&
        (!year || String(student.bynf || "") === year) &&
        (!name || String(student.xm || "").toLowerCase().indexOf(name) >= 0) &&
        (!studentId || String(student.xh || "").toLowerCase().indexOf(studentId) >= 0) &&
        (!college || (String(student.xymc || "") + " " + String(student.xxmc || ""))
          .toLowerCase().indexOf(college) >= 0);
    });
  }

  function taskStudentMaterialHtml(entry) {
    var evidences = (entry.evidences || []).filter(function (item) { return item.evidence_file_id; });
    if (!evidences.length) return '<span class="no-evidence">无佐证材料</span>';
    var destination = (entry.student || {}).byqxlb || "未明确毕业去向";
    var toneClass = " material-kind material-kind--" + materialCategoryTone(destination);
    var visible = evidences.slice(0, 2).map(function (evidence, index) {
      var text = destination + (evidences.length > 1 ? " · " + (index + 1) : "");
      return '<span class="evidence-tag' + toneClass + '" title="' +
        escapeHtml(evidence.file_name || text) + '">' + escapeHtml(text) + '</span>';
    });
    if (evidences.length > 2) {
      visible.push('<span class="evidence-tag evidence-more">+' + (evidences.length - 2) + ' 份</span>');
    }
    return '<div class="evidence-list task-evidence-list">' + visible.join("") + '</div>';
  }

  function renderTaskStudentRow(record) {
    var entry = record.entry;
    var student = entry.student || {};
    var outcome = graduateOutcome(entry);
    return '<tr class="task-student-row" tabindex="0" data-index="' + record.index + '">' +
      '<td><div class="primary-cell"><strong>' + escapeHtml(student.xm || "未关联毕业生") +
      '</strong><small>' + escapeHtml(student.xh || "无学号") + '</small></div></td>' +
      '<td><span class="task-province">' + escapeHtml(student.province || "江苏省") + '</span></td>' +
      '<td><div class="school-cell"><strong>' + escapeHtml(student.xxmc || "未登记学校") +
      '</strong><small>' + escapeHtml(student.xymc || "未登记学院") + '</small></div></td>' +
      '<td>' + escapeHtml(student.bynf || "—") + '</td>' +
      '<td>' + taskStudentMaterialHtml(entry) + '</td>' +
      '<td><span class="status-badge status-badge--' + outcome + '">' + statusLabel(outcome) + '</span></td>' +
      '</tr>';
  }

  function bindTaskStudentRows() {
    document.querySelectorAll(".task-student-row").forEach(function (row) {
      row.addEventListener("click", function () {
        openStudentSummary(Number(row.dataset.index));
      });
      row.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openStudentSummary(Number(row.dataset.index));
        }
      });
    });
  }

  function dimensionOutcome(entry, dimension) {
    var aggregate = ((entry.student || {}).student_result || {});
    if (aggregate.is_complete !== false && aggregate.dimensions && aggregate.dimensions[dimension]) {
      return aggregate.dimensions[dimension];
    }
    var outcomes = (entry.evidences || []).filter(function (item) {
      return item.evidence_file_id;
    }).map(function (evidence) {
      if (dimension === "visual") {
        var sealAssessment = evidence.seal_assessment || {};
        var sealRegions = (evidence.visual_regions || []).filter(function (region) {
          return region.region_type === "seal";
        });
        if (["incomplete", "unclear"].indexOf(sealAssessment.completeness) >= 0) {
          return "suspicious";
        }
        if (["complete", "not_required"].indexOf(sealAssessment.completeness) >= 0) return "pass";
        return sealRegions.length ? "pass" : "unverified";
      }
      var assessment = dimension === "content"
        ? evidence.content_assessment || {}
        : evidence.destination_assessment || {};
      if (dimension === "destination") {
        var hasFieldMismatch = (evidence.field_comparisons || []).some(function (item) {
          return item.status === "mismatch";
        }) || Boolean((evidence.mismatches || []).length);
        if (hasFieldMismatch) return "suspicious";
      }
      if (["questionable", "contradicted", "mismatch", "inconsistent"].indexOf(assessment.verdict) >= 0) {
        return "suspicious";
      }
      if (["supported", "match", "consistent"].indexOf(assessment.verdict) >= 0) return "pass";
      return "unverified";
    });
    if (outcomes.indexOf("suspicious") >= 0) return "suspicious";
    if (outcomes.indexOf("pass") >= 0) return "pass";
    return "unverified";
  }

  function materialDimensionOutcome(evidence, dimension) {
    if (evidence.dimension_outcomes && evidence.dimension_outcomes[dimension]) {
      return evidence.dimension_outcomes[dimension];
    }
    if (dimension === "visual") {
      var sealAssessment = evidence.seal_assessment || {};
      var hasSeal = (evidence.visual_regions || []).some(function (region) {
        return region.region_type === "seal";
      });
      if (["complete", "not_required"].indexOf(sealAssessment.completeness) >= 0 ||
          (!sealAssessment.completeness && hasSeal)) return "pass";
      if (["incomplete", "unclear"].indexOf(sealAssessment.completeness) >= 0) return "suspicious";
      return "unverified";
    }
    var verdict = dimension === "content"
      ? (evidence.content_assessment || {}).verdict
      : (evidence.destination_assessment || {}).verdict;
    if (dimension === "content" && verdict === "supported") return "pass";
    if (dimension === "destination" && verdict === "consistent") return "pass";
    return verdict ? "suspicious" : "unverified";
  }

  function isPrimaryMaterial(evidence, studentResult) {
    if (["review", "fail"].indexOf((studentResult || {}).status) >= 0) return true;
    if (evidence.material_role) return evidence.material_role === "primary";
    var primaryIds = (studentResult || {}).primary_evidence_file_ids || [];
    if (primaryIds.length) {
      return primaryIds.map(String).indexOf(String(evidence.evidence_file_id)) >= 0;
    }
    return evidence.status === "pass" && ["visual", "content", "destination"].every(function (dimension) {
      return materialDimensionOutcome(evidence, dimension) === "pass";
    });
  }

  function failedMaterialDimensions(evidence) {
    var labels = {
      visual: "图像与印章",
      content: "内容完整性",
      destination: "字段一致性"
    };
    return ["visual", "content", "destination"].filter(function (dimension) {
      return materialDimensionOutcome(evidence, dimension) !== "pass";
    }).map(function (dimension) {
      return labels[dimension];
    });
  }

  function uniqueValues(values) {
    return values.filter(function (value, index) {
      return value && values.indexOf(value) === index;
    });
  }

  function renderDrawerDimension(label, outcome, description) {
    return '<article><span>' + escapeHtml(label) + '</span><strong class="dimension-state dimension-state--' +
      outcome + '">' + statusLabel(outcome) + '</strong><p>' + escapeHtml(description) + '</p></article>';
  }

  function renderStudentSummaryDrawer(entry) {
    var student = entry.student || {};
    var outcome = graduateOutcome(entry);
    var studentResult = student.student_result || {};
    var evidences = (entry.evidences || []).filter(function (item) { return item.evidence_file_id; });
    element("studentSummaryDrawerTitle").textContent =
      (student.xm || "未关联毕业生") + " · " + (student.xh || "无学号");
    var materials = evidences.length ? evidences.map(function (evidence) {
      var category = evidence.category || evidence.material_type || "佐证材料";
      var role = isPrimaryMaterial(evidence, studentResult) ? "主要材料" : "次要材料";
      return '<li><span class="material-kind material-kind--' + materialCategoryTone(category) + '">' +
        escapeHtml(category) + '</span><div><strong>' + escapeHtml(evidence.file_name || "未命名材料") +
        '</strong><small>' + role + '</small></div></li>';
    }).join("") : '<li class="drawer-empty-material">该学生没有佐证材料</li>';
    element("studentSummaryDrawerContent").innerHTML =
      '<section class="drawer-result drawer-result--' + outcome + '"><div><span>本次核验结果</span><strong>' +
      statusLabel(outcome) + '</strong></div><p>' +
      escapeHtml(studentResult.summary || (outcome === "unverified" ? "当前没有可用的核验结果" : "已完成材料核验")) +
      '</p></section><dl class="drawer-student-fields"><div><dt>毕业年份</dt><dd>' +
      escapeHtml(student.bynf || "—") + '</dd></div><div><dt>学校</dt><dd>' +
      escapeHtml(student.xxmc || "未登记") + '</dd></div><div><dt>学院</dt><dd>' +
      escapeHtml(student.xymc || "未登记") + '</dd></div><div><dt>毕业去向</dt><dd>' +
      escapeHtml(student.byqxlb || "未登记") + '</dd></div></dl>' +
      '<section class="drawer-section"><header><strong>三项核验概览</strong><span>详细依据请进入 05 页面</span></header><div class="drawer-dimensions">' +
      renderDrawerDimension("印章核验", dimensionOutcome(entry, "visual"), "主要材料中的印章核验结果") +
      renderDrawerDimension("内容完整性", dimensionOutcome(entry, "content"), "主要材料能否完整支持当前就业事实") +
      renderDrawerDimension("毕业去向一致性", dimensionOutcome(entry, "destination"), "主要材料与数据库毕业去向的匹配情况") +
      '</div></section><section class="drawer-section"><header><strong>佐证材料</strong><span>' +
      evidences.length + ' 份</span></header><ul class="drawer-material-list">' + materials + '</ul></section>';
  }

  function setStudentDetailAvailable(available) {
    var button = element("studentDetailNavButton");
    button.classList.toggle("is-disabled", !available);
    button.setAttribute("aria-disabled", available ? "false" : "true");
  }

  function openStudentSummary(index) {
    var entry = state.detailEntries[index];
    if (!entry) return;
    state.detailSelectedIndex = index;
    setStudentDetailAvailable(true);
    renderStudentSummaryDrawer(entry);
    element("studentSummaryDrawer").showModal();
  }

  function openSelectedStudentDetail() {
    if (!state.detailEntries[state.detailSelectedIndex]) return;
    element("studentSummaryDrawer").close();
    renderStudentDetailPage();
    showView("student", true);
  }

  function renderStudentDetailPage() {
    var entry = state.detailEntries[state.detailSelectedIndex];
    if (!entry) {
      element("graduateResultList").innerHTML = '<p class="empty-line">请先从任务学生列表选择一名学生</p>';
      return;
    }
    var student = entry.student || {};
    var outcome = graduateOutcome(entry);
    element("studentDetailHeading").textContent =
      (student.xm || "未关联毕业生") + " · " + (student.xh || "无学号");
    element("studentDetailTaskMeta").textContent =
      (state.currentJob ? state.currentJob.task_name || state.currentJob.job_id : state.detailJobId) +
      " · " + (student.xxmc || "未登记学校") + " / " + (student.xymc || "未登记学院");
    element("studentDetailYear").textContent = student.bynf || "—";
    element("studentDetailDestination").textContent = student.byqxlb || "未登记";
    element("studentDetailStatus").className = "status-badge status-badge--" + outcome;
    element("studentDetailStatus").textContent = statusLabel(outcome);
    element("graduateResultList").innerHTML = renderGraduateResult(entry, state.detailSelectedIndex);
    setStudentDetailAvailable(true);
  }

  function renderGraduateResult(entry, entryIndex) {
    var student = entry.student || {};
    var outcome = graduateOutcome(entry);
    var studentResult = student.student_result || {};
    var aggregateBasis = (studentResult.decision_basis || []).filter(Boolean);
    var aggregateBasisHtml = aggregateBasis.length
      ? '<details class="student-aggregate-basis"><summary>查看总结果判定依据（' + aggregateBasis.length +
        ' 条）</summary><ol>' + aggregateBasis.map(function (item) {
          return '<li>' + escapeHtml(item) + '</li>';
        }).join("") + '</ol></details>'
      : '';
    var evidences = (entry.evidences || []).filter(function (evidence) {
      return evidence.evidence_file_id;
    });
    var primaryMaterials = evidences.filter(function (evidence) {
      return isPrimaryMaterial(evidence, studentResult);
    });
    var secondaryMaterials = evidences.filter(function (evidence) {
      return !isPrimaryMaterial(evidence, studentResult);
    });
    var materialHtml = '';
    if (primaryMaterials.length) {
      var primaryGroupTitle = outcome === "pass" ? "主要材料" : "主要复核材料";
      var groupFailedDimensions = uniqueValues([].concat.apply([], primaryMaterials.map(function (evidence) {
        return failedMaterialDimensions(evidence);
      })));
      var primaryGroupCount = outcome === "pass"
        ? primaryMaterials.length + " 份三项通过"
        : "未通过：" + (groupFailedDimensions.join("、") || "材料综合判定");
      var primaryGroupClass = outcome === "pass"
        ? "material-group--primary"
        : "material-group--failed";
      materialHtml += '<section class="material-group ' + primaryGroupClass + '"><header><div><span>' +
        primaryGroupTitle + '</span><strong>完整核验详情</strong></div><small>' + primaryGroupCount + '</small></header>' +
        primaryMaterials.map(function (evidence, index) {
          return renderMaterialResult(student, evidence, entryIndex + "-primary-" + index);
        }).join("") + '</section>';
    }
    if (secondaryMaterials.length) {
      materialHtml += '<section class="material-group material-group--secondary"><header><div><span>次要材料</span>' +
        '<strong>仅供原件预览</strong></div><small>' + secondaryMaterials.length + ' 份</small></header>' +
        '<div class="secondary-material-list">' + secondaryMaterials.map(renderSecondaryMaterial).join("") +
        '</div></section>';
    }
    if (!evidences.length) {
      materialHtml = '<div class="no-material-result">未提供佐证材料，本次状态为未核查</div>';
    }
    return '<article class="graduate-card">' +
      '<section class="graduate-decision graduate-decision--' + outcome + '">' +
      '<div class="decision-result"><span>本次核验结果</span><strong>' +
      statusLabel(outcome) + '</strong></div><div class="decision-summary"><span>综合判定意见</span><p>' +
      escapeHtml(studentResult.summary || (outcome === "unverified" ? "当前没有可用的核验结果" : "已完成材料核验")) +
      '</p>' + aggregateBasisHtml + '</div></section>' + materialHtml + "</article>";
  }

  function renderSecondaryMaterial(evidence) {
    var ext = fileExtension(evidence.file_name, evidence.extension);
    var previewable = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "pdf"]
      .indexOf(ext) >= 0;
    return '<article class="secondary-material"><div><span>次要材料</span><strong>' +
      escapeHtml(evidence.file_name || "未命名材料") + '</strong><small>' +
      escapeHtml(evidence.category || evidence.material_type || "佐证材料") +
      '</small></div>' + (previewable ? '<button class="mini-button detail-preview-button" type="button" data-id="' +
        evidence.evidence_file_id + '" data-name="' + escapeHtml(evidence.file_name || "") +
        '" data-ext="' + escapeHtml(ext) + '">预览原件</button>' : '<span class="preview-unavailable">暂不支持预览</span>') +
      '</article>';
  }

  function materialOutcome(evidence) {
    if (!evidence.evidence_file_id || ["queued", "running", "failed"].indexOf(evidence.item_status) >= 0) {
      return "unverified";
    }
    if (evidence.status === "pass") return "pass";
    if (evidence.status === "review" || evidence.status === "fail") return "suspicious";
    return "unverified";
  }

  function materialOutcomeLabel(outcome) {
    return ({ pass: "单份有效", suspicious: "结合其他材料", unverified: "未核查" })[outcome] || "未核查";
  }

  function renderMaterialResult(student, evidence, key) {
    if (!evidence.evidence_file_id) {
      return '<div class="no-material-result">未提供佐证材料，无法进行材料核查</div>';
    }
    var panels = buildAuditPanels(student, evidence);
    var failedDimensions = failedMaterialDimensions(evidence);
    var materialFailed = failedDimensions.length > 0;
    var ext = fileExtension(evidence.file_name, evidence.extension);
    var previewable = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "pdf"]
      .indexOf(ext) >= 0;
    var tabs = panels.map(function (panel, index) {
      var selectedPanel = state.activeAuditPanels[key] || panels[0].id;
      var panelState = panel.outcome === "pass"
        ? ""
        : panel.outcome === "unverified" ? " · 未核验" : " · 不通过";
      return '<button class="audit-tab' + (panel.id === selectedPanel ? " is-active" : "") +
        (panel.outcome !== "pass" ? " audit-tab--failed" : "") +
        '" type="button" data-panel="' + escapeHtml(key + "-" + panel.id) + '">' +
        escapeHtml(panel.title + panelState) + "</button>";
    }).join("");
    var panelHtml = panels.map(function (panel, index) {
      var selectedPanel = state.activeAuditPanels[key] || panels[0].id;
      return '<section class="audit-panel' + (panel.id === selectedPanel ? " is-active" : "") +
        '" data-panel-key="' + escapeHtml(key + "-" + panel.id) + '">' +
        panel.html + '</section>';
    }).join("");
    var basis = (evidence.decision_basis || []).filter(Boolean);
    var basisHtml = basis.length
      ? '<details class="material-decision-basis"><summary>判定依据（' + basis.length + ' 条）</summary><ol>' +
        basis.map(function (item) { return '<li>' + escapeHtml(item) + '</li>'; }).join("") +
        '</ol></details>'
      : '';
    var resultBadge = materialFailed
      ? '<span class="material-dimension-status material-dimension-status--fail">未通过：' +
        escapeHtml(failedDimensions.join("、")) + '</span>'
      : '<span class="material-dimension-status material-dimension-status--pass">三项通过</span>';
    return '<section class="material-result' + (materialFailed ? " material-result--fail" : "") + '">' +
      '<header class="material-result-header"><div class="material-result-title"><strong>' +
      escapeHtml(evidence.file_name || "未命名材料") + '</strong><small>' +
      escapeHtml(evidence.category || evidence.material_type || "未识别材料类型") +
      '</small></div><div class="material-result-actions">' + resultBadge +
      (previewable ? '<button class="mini-button detail-preview-button" type="button" data-id="' +
        evidence.evidence_file_id + '" data-name="' + escapeHtml(evidence.file_name) +
        '" data-ext="' + escapeHtml(ext) + '">查看材料</button>' : "") +
      "</div></header>" + basisHtml +
      '<div class="audit-tabs">' + tabs + '</div><div class="audit-panels">' +
      panelHtml + "</div></section>";
  }

  function buildAuditPanels(student, evidence) {
    if (materialOutcome(evidence) === "unverified") {
      return [
        { id: "authenticity", title: "图像与印章", outcome: "unverified", html: emptyEvidenceHtml("尚未完成图像区域核验") },
        { id: "content", title: "内容完整性", outcome: "unverified", html: emptyEvidenceHtml("尚未完成内容核验") },
        { id: "consistency", title: "字段一致性", outcome: "unverified", html: emptyEvidenceHtml("尚未完成一致性核验") }
      ];
    }
    return [
      { id: "authenticity", title: "图像与印章", outcome: materialDimensionOutcome(evidence, "visual"), html: renderVisualEvidence(evidence) },
      { id: "content", title: "内容完整性", outcome: materialDimensionOutcome(evidence, "content"), html: renderContentEvidence(evidence) },
      { id: "consistency", title: "字段一致性", outcome: materialDimensionOutcome(evidence, "destination"), html: renderConsistencyEvidence(student, evidence) }
    ];
  }

  function emptyEvidenceHtml(message) {
    return '<div class="evidence-empty"><strong>暂无可展示依据</strong><p>' +
      escapeHtml(message) + '</p></div>';
  }

  function regionTypeLabel(value) {
    return ({ seal: "印章", signature: "签名", handwriting: "手写内容", photo: "照片",
      document_image: "关键图像", other: "其他区域" })[value] || "图像区域";
  }

  function regionAssessment(value) {
    return ({ no_obvious_anomaly: ["normal", "未见明显异常"], suspicious: ["mismatch", "存在可疑迹象"],
      unclear: ["unknown", "区域不清晰"] })[value] || ["unknown", "待确认"];
  }

  function cleanRegionObservation(value) {
    return String(value || "")
      .replace(/[，,\s]*(no_obvious_anomaly|suspicious|unclear)\s*$/i, "")
      .trim();
  }

  function isGenericSealText(value) {
    var text = String(value || "").replace(/[，。；;\s]/g, "");
    return !text || [
      "检测到印章", "已检测到印章", "不检查印章内容和外观",
      "仅记录印章存在", "仅记录印章是否存在",
      "仅记录印章存在不检查章面内容或外观"
    ].indexOf(text) >= 0;
  }

  function regionLocation(region) {
    var box = region.bbox || {};
    var centerX = (Number(box.x1 || 0) + Number(box.x2 || 0)) / 2;
    var centerY = (Number(box.y1 || 0) + Number(box.y2 || 0)) / 2;
    var horizontal = centerX < 340 ? "左侧" : centerX > 660 ? "右侧" : "中部";
    var vertical = centerY < 340 ? "上方" : centerY > 660 ? "下方" : "中部";
    return "第 " + (region.page_number || 1) + " 页" + vertical + horizontal;
  }

  function sealRegionText(region, field) {
    var value = field === "observation"
      ? cleanRegionObservation(region.observation)
      : String(region.basis || "").trim();
    if (!isGenericSealText(value)) return value;
    return regionLocation(region) + (field === "observation"
      ? "定位到一枚实际印章"
      : "可确认存在盖章痕迹");
  }

  function regionCropUrl(evidenceId, region) {
    var box = region.bbox || {};
    var params = new URLSearchParams({
      x1: String(box.x1 || 0), y1: String(box.y1 || 0),
      x2: String(box.x2 || 0), y2: String(box.y2 || 0)
    });
    return API_ROOT + "/evidence-files/" + encodeURIComponent(evidenceId) +
      "/regions/" + encodeURIComponent(region.page_number || 1) + "?" + params.toString();
  }

  function renderVisualEvidence(evidence) {
    var regionPriority = {
      seal: 0,
      signature: 1,
      handwriting: 2,
      photo: 3,
      document_image: 4,
      other: 5
    };
    var regions = (evidence.visual_regions || []).slice().sort(function (left, right) {
      var leftPriority = Object.prototype.hasOwnProperty.call(regionPriority, left.region_type)
        ? regionPriority[left.region_type] : 99;
      var rightPriority = Object.prototype.hasOwnProperty.call(regionPriority, right.region_type)
        ? regionPriority[right.region_type] : 99;
      return leftPriority - rightPriority || Number(left.page_number || 1) - Number(right.page_number || 1);
    });
    var sealAssessment = evidence.seal_assessment || {};
    var sealState = ({ complete: ["normal", "印章已确认"], incomplete: ["mismatch", "印章需复核"],
      not_required: ["normal", "无需印章"], unclear: ["unknown", "印章需复核"] })[
      sealAssessment.completeness
    ] || ["unknown", "印章待确认"];
    var regionHtml = regions.map(function (region) {
      var assessment = regionAssessment(region.assessment);
      var image = evidence.evidence_file_id && region.bbox
        ? '<img loading="lazy" src="' + escapeHtml(regionCropUrl(evidence.evidence_file_id, region)) +
          '" alt="第' + escapeHtml(region.page_number || 1) + '页' + escapeHtml(regionTypeLabel(region.region_type)) + '裁剪区域">'
        : '<div class="region-image-placeholder">区域图像不可用</div>';
      var isSeal = region.region_type === "seal";
      var visibleText = String(region.visible_text || "").trim();
      var sealRows = '<div><dt>所在位置</dt><dd>' + escapeHtml(sealRegionText(region, "observation")) +
        '</dd></div>' + (visibleText ? '<div><dt>可辨文字</dt><dd>' + escapeHtml(visibleText) +
        '</dd></div>' : '') + '<div><dt>核验判断</dt><dd>' +
        escapeHtml(sealRegionText(region, "basis")) + '</dd></div>';
      var otherRows = '<div><dt>可见文字</dt><dd>' +
        escapeHtml(region.visible_text || "未识别到清晰文字") +
        '</dd></div><div><dt>区域观察</dt><dd>' +
        escapeHtml(cleanRegionObservation(region.observation) || "无补充描述") +
        '</dd></div><div><dt>判定依据</dt><dd>' +
        escapeHtml(region.basis || "未提供区域依据") + '</dd></div>';
      return '<article class="visual-region-card">' +
        '<div class="visual-region-image">' + image + '</div><div class="visual-region-copy">' +
        '<div class="region-heading"><span>第 ' + escapeHtml(region.page_number || 1) + ' 页 · ' +
        escapeHtml(regionTypeLabel(region.region_type)) + '</span><b class="evidence-state evidence-state--' +
        assessment[0] + '">' + (isSeal ? "已定位" : assessment[1]) + '</b></div>' +
        '<dl>' + (isSeal ? sealRows : otherRows) + '</dl></div></article>';
    }).join("");
    return '<div class="panel-intro"><div><strong>材料关键区域</strong><p>' +
      escapeHtml(sealAssessment.basis || "展示模型在原始页定位的印章、签名和手写区域") +
      '</p></div><span class="evidence-state evidence-state--' + sealState[0] + '">' +
      sealState[1] + '</span></div>' +
      (regionHtml ? '<div class="visual-region-grid">' + regionHtml + '</div>' :
        emptyEvidenceHtml("模型未定位到可展示的印章、签名或手写区域。"));
  }

  function evidencePointState(value) {
    return value === "supports" || value === "supported" ? "match" :
      value === "contradicts" || value === "contradicted" ? "mismatch" : "unknown";
  }

  function negativeEvidenceText(value) {
    var text = String(value || "")
      .replace(/未发现明显异常|未见明显异常|未发现异常|未见异常|不存在明确冲突|无明确冲突/g, "");
    return /未(体现|填写|提供|显示|载明|注明|识别|找到|发现|确认|签署|盖章)|没有|无任何|缺少|缺失|为空|空白|不足以|无法(证明|确认|判断|建立|核对|支持)|不构成|不能证明|尚未(录用|聘用|就业|接收|入学|入站|签约)/.test(text);
  }

  function advisoryEvidenceText(value) {
    return /工作地点|学习地点|所在地|地址|专业|学院|毕业年份|毕业日期|签约日期|录取日期|报到日期|日期|单位性质|行业|岗位|职位|薪酬|工资|联系人|联系电话|电话|印章|盖章|签名|签字/.test(String(value || ""));
  }

  function contentPointOutcome(item) {
    var explicit = item.decision_assessment;
    if (explicit) return explicit;
    var original = item.assessment || "unclear";
    var detail = String(item.claim || "") + " " + String(item.observed_text || "");
    if (advisoryEvidenceText(detail) && (original === "contradicts" || negativeEvidenceText(detail))) return "advisory";
    if (original === "supports" && negativeEvidenceText(detail)) return "contradicts";
    return original;
  }

  function ruleOutcome(item) {
    var explicit = item.decision_result;
    if (explicit) return explicit;
    var original = item.result || "not_observed";
    var detail = String(item.rule || "") + " " + String(item.evidence || "");
    if (item.importance !== "essential" && (original === "contradicted" || negativeEvidenceText(detail))) return "advisory";
    if (original === "supported" && negativeEvidenceText(detail)) return "contradicted";
    return original;
  }

  function contentOutcomeLabel(value) {
    return ({ supports: ["match", "符合要求"], supported: ["match", "符合要求"],
      contradicts: ["mismatch", "不符合要求"], contradicted: ["mismatch", "不符合要求"],
      advisory: ["advisory", "提示项"], unclear: ["unknown", "无法判断"],
      not_observed: ["unknown", "未观察到"] })[value] || ["unknown", "无法判断"];
  }

  function renderContentEvidence(evidence) {
    var assessment = evidence.content_assessment || {};
    var verdict = assessment.verdict === "supported" ? ["match", "内容完整"] :
      assessment.verdict === "questionable" ? ["mismatch", "内容不符合要求"] : ["unknown", "关键证据不足"];
    var points = (assessment.evidence_points || []).map(function (item) {
      var resultValue = contentPointOutcome(item);
      var pointResult = contentOutcomeLabel(resultValue);
      return '<article class="evidence-point evidence-point--' + pointResult[0] + '"><span class="page-chip">第 ' +
        escapeHtml(item.page_number || "?") + ' 页</span><div><strong>' + escapeHtml(item.claim || "内容判断") +
        '</strong><blockquote>' + escapeHtml(item.observed_text || "未提供原文") + '</blockquote></div>' +
        '<b class="evidence-state evidence-state--' + pointResult[0] + '">' + pointResult[1] + '</b></article>';
    });
    if (!points.length) {
      (evidence.rule_results || []).forEach(function (item) {
        if (["supported", "contradicted", "not_observed", "pass", "fail"].indexOf(item.result) < 0) return;
        var resultValue = ruleOutcome(item);
        var pointResult = contentOutcomeLabel(resultValue);
        points.push('<article class="evidence-point evidence-point--' + pointResult[0] + '"><span class="page-chip">' +
          (item.page_number ? "第 " + escapeHtml(item.page_number) + " 页" : "材料") + '</span><div><strong>' +
          escapeHtml(item.rule || "类别规则") + '</strong><blockquote>' + escapeHtml(item.evidence || "未提供原文") +
          '</blockquote></div><b class="evidence-state evidence-state--' + pointResult[0] + '">' +
          pointResult[1] + '</b></article>');
      });
    }
    return '<div class="panel-intro"><div><strong>内容完整性结论</strong><p>' +
      escapeHtml(assessment.summary || evidence.summary || "根据材料中的结果性事实进行判断") +
      '</p></div><span class="verdict-badge verdict-badge--' + verdict[0] + '">' + verdict[1] + '</span></div>' +
      (points.length ? '<div class="evidence-point-list">' + points.join("") + '</div>' :
        emptyEvidenceHtml("没有可追溯的内容判断依据，建议重新核验或人工查看原始材料。"));
  }

  function comparisonStatus(value) {
    return ({ match: ["match", "一致"], compatible: ["match", "可合理对应"],
      mismatch: ["mismatch", "不一致"], not_observed: ["unknown", "材料未识别"],
      extracted_only: ["extracted", "仅材料提取"], not_comparable: ["unknown", "无法比较"],
      advisory_difference: ["advisory", "提示差异"] })[value] || ["unknown", "待确认"];
  }

  function renderConsistencyEvidence(student, evidence) {
    var destination = evidence.destination_assessment || {};
    var destinationState = destination.verdict === "consistent" ? ["match", "去向一致"] :
      destination.verdict === "inconsistent" ? ["mismatch", "去向不一致"] : ["unknown", "待确认"];
    var destinationPoints = (destination.evidence_points || []).map(function (item) {
      return '<li><span>第 ' + escapeHtml(item.page_number || "?") + ' 页</span><strong>' +
        escapeHtml(item.observed_text || "未提供原文") + '</strong><p>' +
        escapeHtml(item.explanation || "") + '</p></li>';
    }).join("");
    var comparisons = evidence.field_comparisons || [];
    var rows = comparisons.map(function (item) {
      var statusValue = comparisonStatus(item.status);
      return '<tr class="comparison-row comparison-row--' + statusValue[0] + '"><td><strong>' +
        escapeHtml(item.label || item.field || "字段") + '</strong><small>' +
        escapeHtml(item.database_field || "") + '</small></td><td>' +
        escapeHtml(item.database_value || "未登记") + '</td><td><strong>' +
        escapeHtml(item.model_value || "未识别") + '</strong>' +
        (item.observed_text ? '<small>原文：' + escapeHtml(item.observed_text) + '</small>' : '') +
        '</td><td><span class="comparison-badge comparison-badge--' + statusValue[0] + '">' +
        statusValue[1] + '</span><small>' + escapeHtml(item.basis || "") + '</small></td></tr>';
    }).join("");
    if (!rows && evidence.mismatches && evidence.mismatches.length) {
      rows = evidence.mismatches.map(function (item) {
        return '<tr class="comparison-row comparison-row--mismatch"><td><strong>' +
          escapeHtml(item.field || "字段") + '</strong></td><td>' + escapeHtml(item.registered_value || "未登记") +
          '</td><td>' + escapeHtml(item.evidence_value || "未识别") +
          '</td><td><span class="comparison-badge comparison-badge--mismatch">不一致</span><small>' +
          escapeHtml(item.description || "") + '</small></td></tr>';
      }).join("");
    }
    return '<div class="destination-card destination-card--' + destinationState[0] + '"><div><span>毕业去向一致性</span><strong>' +
      destinationState[1] + '</strong></div><dl><div><dt>数据库登记</dt><dd>' +
      escapeHtml(destination.registered_destination || student.byqxlb || "未登记") +
      '</dd></div><div><dt>材料识别</dt><dd>' + escapeHtml(destination.recognized_destination || evidence.material_type || "未识别") +
      '</dd></div></dl><p>' + escapeHtml(destination.summary || "根据材料类型与登记毕业去向进行业务含义比对") + '</p>' +
      (destinationPoints ? '<ul class="destination-evidence">' + destinationPoints + '</ul>' : '') + '</div>' +
      '<div class="comparison-heading"><div><strong>数据库字段逐项比对</strong><p>核心身份和单位字段决定结论；地点、日期、专业等辅助字段差异仅作提示。</p></div><span class="count-badge">' +
      comparisons.length + ' 个字段</span></div>' +
      (rows ? '<div class="comparison-table-wrap"><table class="comparison-table"><thead><tr><th>字段</th><th>数据库值</th><th>模型识别值</th><th>比对结果与依据</th></tr></thead><tbody>' +
        rows + '</tbody></table></div>' : emptyEvidenceHtml("数据库和材料中暂时没有可进行逐项核对的字段。"));
  }

  function renderDetailPagination(total, totalPages) {
    var pagination = element("detailPagination");
    pagination.classList.toggle("hidden", total === 0);
    if (!total) return;
    var first = (state.detailPage - 1) * state.detailPageSize + 1;
    var last = Math.min(state.detailPage * state.detailPageSize, total);
    element("detailPageSummary").textContent =
      "第 " + state.detailPage + " / " + totalPages + " 页 · " + first + "–" + last +
      " / " + total + " 名 · 每页 " + state.detailPageSize + " 名";
    element("detailFirstPage").disabled = state.detailPage <= 1;
    element("detailPreviousPage").disabled = state.detailPage <= 1;
    element("detailNextPage").disabled = state.detailPage >= totalPages;
    element("detailLastPage").disabled = state.detailPage >= totalPages;

    var start = Math.max(1, state.detailPage - 2);
    var end = Math.min(totalPages, start + 4);
    start = Math.max(1, end - 4);
    var buttons = [];
    for (var page = start; page <= end; page += 1) {
      buttons.push('<button class="page-number detail-page-number' + (page === state.detailPage ? " is-active" : "") +
        '" type="button" data-page="' + page + '">' + page + "</button>");
    }
    element("detailPageNumbers").innerHTML = buttons.join("");
    document.querySelectorAll(".detail-page-number").forEach(function (button) {
      button.addEventListener("click", function () {
        changeDetailPage(Number(button.dataset.page));
      });
    });
  }

  function changeDetailPage(page) {
    var totalPages = Math.max(1, Math.ceil(filteredTaskStudentRecords().length / state.detailPageSize));
    var target = Math.min(Math.max(1, page), totalPages);
    if (target === state.detailPage) return;
    state.detailPage = target;
    renderDetailPage();
  }

  function renderPreviewSlide() {
    var item = state.previewItems[state.previewIndex];
    if (!item) return;
    var dialog = element("previewDialog");
    var ext = fileExtension(item.file_name, item.extension);
    var url = API_ROOT + "/evidence-files/" + encodeURIComponent(item.evidence_file_id) + "/preview";
    element("previewTitle").textContent = item.file_name || ("材料 #" + item.evidence_file_id);
    element("previewCounter").textContent = (state.previewIndex + 1) + " / " + state.previewItems.length;
    element("previewPreviousButton").disabled = state.previewIndex <= 0;
    element("previewNextButton").disabled = state.previewIndex >= state.previewItems.length - 1;
    element("previewBody").innerHTML = ext === "pdf"
      ? '<iframe title="PDF材料预览" src="' + url + '"></iframe>'
      : '<img alt="' + escapeHtml(item.file_name || "材料预览") + '" src="' + url + '">';
    return dialog;
  }

  function openGraduateEvidencePreview(graduateId, evidenceId) {
    var graduate = state.graduates.find(function (item) {
      return item.graduate_id === graduateId;
    });
    if (!graduate || !(graduate.evidences || []).length) return;
    state.previewItems = graduate.evidences.filter(function (item) {
      return item.preview_supported;
    });
    state.previewIndex = Math.max(0, state.previewItems.findIndex(function (item) {
      return Number(item.evidence_file_id) === Number(evidenceId);
    }));
    var dialog = renderPreviewSlide();
    if (dialog) dialog.showModal();
  }

  function openPreview(id, name, extension) {
    state.previewItems = [{ evidence_file_id: id, file_name: name, extension: extension }];
    state.previewIndex = 0;
    var dialog = renderPreviewSlide();
    dialog.showModal();
  }

  function downloadBlob(name, content, type) {
    var blob = new Blob([content], { type: type });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function exportJson() {
    if (!state.currentResults) return;
    downloadBlob(
      state.currentResults.job_id + ".json",
      JSON.stringify(withoutConfidenceFields(state.currentResults), null, 2),
      "application/json;charset=utf-8"
    );
  }

  function withoutConfidenceFields(value) {
    if (Array.isArray(value)) {
      return value.map(withoutConfidenceFields);
    }
    if (!value || typeof value !== "object") return value;
    return Object.keys(value).reduce(function (cleaned, key) {
      if (key.toLowerCase().indexOf("confidence") >= 0) return cleaned;
      cleaned[key] = withoutConfidenceFields(value[key]);
      return cleaned;
    }, {});
  }

  function exportCsv() {
    if (!state.currentResults) return;
    var columns = [
      "task_name", "job_id", "graduate_id", "xh", "xm", "xxmc", "xymc", "bynf",
      "byqxlb", "graduate_status", "evidence_file_id", "file_name", "category",
      "material_status", "material_type", "summary", "material_summary",
      "content_verdict", "destination_verdict", "visual_region_count",
      "decision_basis", "material_decision_basis", "field_comparisons"
    ];
    var rows = [];
    state.detailEntries.forEach(function (entry) {
      var student = entry.student || {};
      var studentResult = student.student_result || {};
      (entry.evidences || []).forEach(function (evidence) {
        rows.push({
          task_name: state.currentResults.task_name || "",
          job_id: state.currentResults.job_id,
          graduate_id: student.graduate_id || "",
          xh: student.xh || "",
          xm: student.xm || "",
          xxmc: student.xxmc || "",
          xymc: student.xymc || "",
          bynf: student.bynf || "",
          byqxlb: student.byqxlb || "",
          graduate_status: statusLabel(graduateOutcome(entry)),
          evidence_file_id: evidence.evidence_file_id || "",
          file_name: evidence.file_name || "",
          category: evidence.category || "",
          material_status: statusLabel(materialOutcome(evidence)),
          material_type: evidence.material_type || "",
          summary: studentResult.summary || "",
          material_summary: evidence.summary || "",
          content_verdict: (evidence.content_assessment || {}).verdict || "",
          destination_verdict: (evidence.destination_assessment || {}).verdict || "",
          visual_region_count: (evidence.visual_regions || []).length,
          decision_basis: JSON.stringify(studentResult.decision_basis || []),
          material_decision_basis: JSON.stringify(evidence.decision_basis || []),
          field_comparisons: JSON.stringify(evidence.field_comparisons || [])
        });
      });
    });
    var csv = [columns.join(",")].concat(rows.map(function (row) {
      return columns.map(function (column) {
        return '"' + String(row[column] == null ? "" : row[column]).replace(/"/g, '""') + '"';
      }).join(",");
    })).join("\r\n");
    downloadBlob(
      state.currentResults.job_id + "-summary.csv",
      "\ufeff" + csv,
      "text/csv;charset=utf-8"
    );
  }

  function resetFilters() {
    [
      "provinceFilter", "yearFilter", "schoolFilter", "collegeFilter",
      "nameFilter", "studentIdFilter", "materialCategoryFilter"
    ].forEach(function (id) {
      element(id).value = "";
    });
    element("evidenceStatusFilter").value = "all";
    element("verificationStatusFilter").value = "all";
    state.dataPage = 1;
    loadGraduates();
  }

  function bindEvents() {
    document.querySelectorAll("[data-view]").forEach(function (button) {
      button.addEventListener("click", function () {
        if (button.dataset.view === "student") {
          if (!state.detailEntries[state.detailSelectedIndex] ||
              button.getAttribute("aria-disabled") === "true") {
            showToast("请先在 04 任务学生列表中选择一名学生", true);
            return;
          }
          renderStudentDetailPage();
        }
        showView(button.dataset.view, true);
      });
    });
    element("mobileSidebarButton").addEventListener("click", function () {
      element("appSidebar").classList.toggle("is-open");
    });
    window.addEventListener("popstate", function () {
      showView(viewFromLocation(), false);
    });
    element("graduateFilterForm").addEventListener("submit", function (event) {
      event.preventDefault();
      state.dataPage = 1;
      loadGraduates();
    });
    element("resetFilterButton").addEventListener("click", resetFilters);
    element("refreshDataButton").addEventListener("click", function () {
      loadGraduates();
      loadOverview();
    });
    element("dataPreviousPage").addEventListener("click", function () {
      if (state.dataPage > 1) {
        state.dataPage -= 1;
        loadGraduates();
      }
    });
    element("dataNextPage").addEventListener("click", function () {
      if (state.dataPage < state.dataTotalPages) {
        state.dataPage += 1;
        loadGraduates();
      }
    });
    element("selectPageCheckbox").addEventListener("change", function (event) {
      state.graduates.forEach(function (graduate) {
        if (event.target.checked) {
          state.selectedGraduates.set(graduate.graduate_id, graduate);
        } else {
          state.selectedGraduates.delete(graduate.graduate_id);
        }
      });
      if (state.graduatePageData) renderGraduateTable(state.graduatePageData);
    });
    element("selectFilteredButton").addEventListener("click", selectAllFilteredGraduates);
    element("clearSelectionButton").addEventListener("click", clearSelection);
    element("goCreateTaskButton").addEventListener("click", function () {
      if (!state.selectedGraduates.size) {
        showToast("请先选择需要核验的毕业生", true);
        return;
      }
      showView("create", true);
    });
    element("taskNameInput").addEventListener("input", updateCreateButton);
    element("createTaskForm").addEventListener("submit", function (event) {
      event.preventDefault();
      createTask();
    });
    element("liveViewDetailButton").addEventListener("click", function () {
      if (state.currentJob) loadJobDetail(state.currentJob.job_id, true);
    });
    element("liveCancelTaskButton").addEventListener("click", function () {
      if (state.currentJob) {
        openCancelDialog(state.currentJob.job_id, state.currentJob.task_name);
      }
    });
    element("jobSearchForm").addEventListener("submit", function (event) {
      event.preventDefault();
      loadJobs();
    });
    element("refreshJobsButton").addEventListener("click", loadJobs);
    element("summaryFilterForm").addEventListener("submit", function (event) {
      event.preventDefault();
      state.summaryPage = 1;
      loadCheckedStudentSummary();
    });
    element("resetSummaryFilterButton").addEventListener("click", resetSummaryFilters);
    element("refreshSummaryButton").addEventListener("click", loadCheckedStudentSummary);
    element("refreshParameterButton").addEventListener("click", loadModelParameters);
    element("checkedStudentTableBody").addEventListener("click", function (event) {
      var button = event.target.closest(".summary-detail-button");
      if (!button) return;
      button.disabled = true;
      loadJobDetail(button.dataset.jobId, false, button.dataset.graduateId)
        .finally(function () { button.disabled = false; });
    });
    element("summaryFirstPage").addEventListener("click", function () {
      changeSummaryPage(1);
    });
    element("summaryPreviousPage").addEventListener("click", function () {
      changeSummaryPage(state.summaryPage - 1);
    });
    element("summaryNextPage").addEventListener("click", function () {
      changeSummaryPage(state.summaryPage + 1);
    });
    element("summaryLastPage").addEventListener("click", function () {
      changeSummaryPage(state.summaryTotalPages);
    });
    element("summaryPageNumbers").addEventListener("click", function (event) {
      var button = event.target.closest(".summary-page-number");
      if (button) changeSummaryPage(Number(button.dataset.page));
    });
    element("taskStudentFilterForm").addEventListener("submit", function (event) {
      event.preventDefault();
      state.detailPage = 1;
      renderDetailPage();
    });
    element("resetTaskStudentFilterButton").addEventListener("click", function () {
      resetTaskStudentFilters(true);
    });
    element("cancelDeleteJobButton").addEventListener("click", function () {
      state.pendingDeleteJob = null;
      element("deleteJobDialog").close();
    });
    element("confirmDeleteJobButton").addEventListener("click", deletePendingJob);
    element("deleteJobDialog").addEventListener("click", function (event) {
      if (event.target === element("deleteJobDialog")) {
        state.pendingDeleteJob = null;
        element("deleteJobDialog").close();
      }
    });
    element("dismissCancelJobButton").addEventListener("click", function () {
      state.pendingCancelJob = null;
      element("cancelJobDialog").close();
    });
    element("confirmCancelJobButton").addEventListener("click", cancelPendingJob);
    element("cancelJobDialog").addEventListener("click", function (event) {
      if (event.target === element("cancelJobDialog")) {
        state.pendingCancelJob = null;
        element("cancelJobDialog").close();
      }
    });
    element("backToJobsButton").addEventListener("click", function () {
      showView("jobs", true);
    });
    element("backToTaskStudentsButton").addEventListener("click", function () {
      showView("detail", true);
    });
    element("closeStudentSummaryDrawer").addEventListener("click", function () {
      element("studentSummaryDrawer").close();
    });
    element("openStudentDetailButton").addEventListener("click", openSelectedStudentDetail);
    element("studentSummaryDrawer").addEventListener("click", function (event) {
      if (event.target === element("studentSummaryDrawer")) {
        element("studentSummaryDrawer").close();
      }
    });
    element("exportJsonButton").addEventListener("click", exportJson);
    element("exportCsvButton").addEventListener("click", exportCsv);
    element("detailFirstPage").addEventListener("click", function () {
      changeDetailPage(1);
    });
    element("detailPreviousPage").addEventListener("click", function () {
      changeDetailPage(state.detailPage - 1);
    });
    element("detailNextPage").addEventListener("click", function () {
      changeDetailPage(state.detailPage + 1);
    });
    element("detailLastPage").addEventListener("click", function () {
      changeDetailPage(Math.ceil(filteredTaskStudentRecords().length / state.detailPageSize));
    });
    element("graduateResultList").addEventListener("click", function (event) {
      var tab = event.target.closest(".audit-tab");
      if (tab) {
        var material = tab.closest(".material-result");
        var panelParts = String(tab.dataset.panel || "").split("-");
        panelParts.pop();
        state.activeAuditPanels[panelParts.join("-")] = String(tab.dataset.panel || "").split("-").pop();
        material.querySelectorAll(".audit-tab").forEach(function (item) {
          item.classList.toggle("is-active", item === tab);
        });
        material.querySelectorAll(".audit-panel").forEach(function (panel) {
          panel.classList.toggle("is-active", panel.dataset.panelKey === tab.dataset.panel);
        });
      }
      var preview = event.target.closest(".detail-preview-button");
      if (preview) openPreview(preview.dataset.id, preview.dataset.name, preview.dataset.ext);
    });
    element("closePreviewButton").addEventListener("click", function () {
      element("previewDialog").close();
      element("previewBody").innerHTML = "";
    });
    element("previewPreviousButton").addEventListener("click", function () {
      if (state.previewIndex > 0) {
        state.previewIndex -= 1;
        renderPreviewSlide();
      }
    });
    element("previewNextButton").addEventListener("click", function () {
      if (state.previewIndex < state.previewItems.length - 1) {
        state.previewIndex += 1;
        renderPreviewSlide();
      }
    });
    element("previewDialog").addEventListener("click", function (event) {
      if (event.target === element("previewDialog")) {
        element("previewDialog").close();
        element("previewBody").innerHTML = "";
      }
    });
  }

  async function initialize() {
    bindEvents();
    showView(viewFromLocation(), false);
    await loadHealth();
    await Promise.allSettled([loadOverview(), loadFilterOptions()]);
    await Promise.allSettled([loadGraduates(), loadJobs()]);
  }

  initialize();
})();
