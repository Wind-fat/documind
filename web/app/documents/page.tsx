"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import PageHeader from "@/components/PageHeader";
import PageViewer from "@/components/PageViewer";
import { EmptyState, ErrorState, LoadingState } from "@/components/States";
import {
  getOutline,
  listDocuments,
  toApiError,
  uploadDocument,
  type ApiError,
} from "@/lib/api";
import { dateTime, pageLabel } from "@/lib/format";
import { useCurrentUser } from "@/lib/session";
import type { DocumentMeta, Outline, UploadResult, ViewerTarget } from "@/lib/types";

const ROLE_OPTIONS = ["all", "employee", "finance", "admin"];

export default function DocumentsPage() {
  const user = useCurrentUser();
  const [docs, setDocs] = useState<DocumentMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const [selected, setSelected] = useState<DocumentMeta | null>(null);
  const [outline, setOutline] = useState<Outline | null>(null);
  const [outlineLoading, setOutlineLoading] = useState(false);
  const [outlineError, setOutlineError] = useState<ApiError | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [roles, setRoles] = useState<string[]>(["all"]);
  const [extraRoles, setExtraRoles] = useState("");
  const [token, setToken] = useState("dev-admin-token");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<ApiError | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [viewer, setViewer] = useState<ViewerTarget | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    listDocuments()
      .then((data) => {
        setDocs(data);
        setLoading(false);
      })
      .catch((err: unknown) => {
        setError(toApiError(err));
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    load();
  }, [load, user]);

  // 切身份后旧选区可能已不可见
  useEffect(() => {
    setSelected(null);
    setOutline(null);
  }, [user]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    setOutlineLoading(true);
    setOutlineError(null);
    setOutline(null);
    getOutline(selected.id, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        setOutline(data);
        setOutlineLoading(false);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setOutlineError(toApiError(err));
        setOutlineLoading(false);
      });
    return () => controller.abort();
  }, [selected]);

  const aclRoles = useMemo(() => {
    const extra = extraRoles
      .split(/[,，\s]+/)
      .map((item) => item.trim())
      .filter(Boolean);
    return Array.from(new Set([...roles, ...extra]));
  }, [roles, extraRoles]);

  const toggleRole = (role: string): void => {
    setRoles((prev) => (prev.includes(role) ? prev.filter((r) => r !== role) : [...prev, role]));
  };

  const submitUpload = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    if (!file) {
      setUploadError(toApiError(new Error("请先选择要上传的文件（pdf / docx / md）")));
      return;
    }
    if (aclRoles.length === 0) {
      setUploadError(toApiError(new Error("acl_roles 不能为空，至少选一个角色")));
      return;
    }
    setUploading(true);
    setUploadError(null);
    setUploadResult(null);
    try {
      const result = await uploadDocument(file, aclRoles, title, token);
      setUploadResult(result);
      setFile(null);
      setTitle("");
      if (fileInput.current) fileInput.current.value = "";
      load();
    } catch (err) {
      setUploadError(toApiError(err));
    } finally {
      setUploading(false);
    }
  };

  return (
    <>
      <PageHeader
        title="文档库"
        subtitle={`${docs.length} 篇文档 · 上传需 X-Admin-Token`}
        right={
          <>
            <span>Bearer · {user}</span>
            <button type="button" className="btn btn-sm" onClick={load} disabled={loading}>
              刷新
            </button>
          </>
        }
      />

      <div className="page">
        <div className="page-narrow">
          <div className="docs-grid">
            <div className="card" style={{ overflow: "hidden" }}>
              <div className="card-head">
                <span className="card-title">文档列表</span>
                <span className="card-sub">visible 由 ACL × 当前身份决定</span>
              </div>

              {loading ? <LoadingState label="读取 /documents…" rows={4} /> : null}
              {error ? <div style={{ padding: 14 }}><ErrorState message={error.detail} status={error.status} onRetry={load} /></div> : null}

              {!loading && !error && docs.length === 0 ? (
                <EmptyState
                  icon="▤"
                  title="还没有文档"
                  desc="在右侧用 admin token 上传第一份文档，后端会解析、切块并索引。"
                />
              ) : null}

              {!loading && !error && docs.length > 0 ? (
                <table className="doc-table">
                  <thead>
                    <tr>
                      <th>文档</th>
                      <th>页数</th>
                      <th>ACL</th>
                      <th>当前身份可见</th>
                      <th>创建时间</th>
                    </tr>
                  </thead>
                  <tbody>
                    {docs.map((doc) => {
                      const isSelected = selected?.id === doc.id;
                      return (
                        <tr
                          key={doc.id}
                          className={isSelected ? "selected" : ""}
                          style={{ cursor: doc.visible ? "pointer" : "not-allowed", opacity: doc.visible ? 1 : 0.55 }}
                          onClick={() => {
                            if (!doc.visible) return;
                            setSelected((prev) => (prev?.id === doc.id ? null : doc));
                          }}
                        >
                          <td>
                            {/* P1-14：整行 onClick 只有鼠标可达。真正的可聚焦控件是标题按钮，
                                Tab 能停在它上面、Enter/Space 打开章节树，读屏也知道这是可展开项。 */}
                            <button
                              type="button"
                              className="doc-pick"
                              aria-expanded={isSelected}
                              aria-disabled={!doc.visible}
                              title={doc.visible ? "加载/收起章节树" : "当前身份无权访问该文档"}
                              onClick={(event) => {
                                // 行上也有 onClick（鼠标点空白处仍可选）；这里拦掉，避免切换两次
                                event.stopPropagation();
                                if (!doc.visible) return;
                                setSelected((prev) => (prev?.id === doc.id ? null : doc));
                              }}
                            >
                              <span className="doc-name">{doc.title}</span>
                              <span className="doc-file">
                                {doc.filename} · <span className="doc-id">{doc.id}</span>
                              </span>
                            </button>
                          </td>
                          <td className="mono">{doc.pages}</td>
                          <td>
                            {doc.acl_roles.length > 0 ? (
                              doc.acl_roles.map((role) => (
                                <span key={role} className="acl-chip">
                                  {role}
                                </span>
                              ))
                            ) : (
                              <span className="muted">—</span>
                            )}
                          </td>
                          <td>
                            {doc.visible ? (
                              <span className="tag ok">可见</span>
                            ) : (
                              <span className="tag danger" title="后端对无权文档不返回内容，也不返回 404 细节">
                                无权访问
                              </span>
                            )}
                          </td>
                          <td className="mono muted" style={{ fontSize: 11 }}>
                            {dateTime(doc.created_at)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              ) : null}
            </div>

            <div style={{ display: "grid", gap: 16 }}>
              <div className="card">
                <div className="card-head">
                  <span className="card-title">
                    上传文档
                    <span className="tag warn">admin</span>
                  </span>
                  <span className="card-sub">POST /documents</span>
                </div>
                <form className="upload-form" onSubmit={(event) => void submitUpload(event)}>
                  <label className={`file-drop${file ? " has-file" : ""}`}>
                    <input
                      ref={fileInput}
                      type="file"
                      accept=".pdf,.docx,.md,.txt"
                      onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                    />
                    {file ? `${file.name} · ${(file.size / 1024).toFixed(0)} KB` : "点击选择文件 · pdf / docx / md"}
                  </label>

                  <div className="field">
                    <label htmlFor="doc-title">标题（可选，默认取文件名）</label>
                    <input
                      id="doc-title"
                      className="input"
                      value={title}
                      onChange={(event) => setTitle(event.target.value)}
                      placeholder="例如：公司章程"
                    />
                  </div>

                  <div className="field">
                    <label>acl_roles（决定哪些角色可见）</label>
                    <div className="check-row">
                      {ROLE_OPTIONS.map((role) => (
                        <label key={role} className="check">
                          <input type="checkbox" checked={roles.includes(role)} onChange={() => toggleRole(role)} />
                          {role}
                        </label>
                      ))}
                    </div>
                    <input
                      className="input"
                      value={extraRoles}
                      onChange={(event) => setExtraRoles(event.target.value)}
                      placeholder="追加自定义角色，逗号分隔（可选）"
                    />
                    <span className="card-sub mono">{JSON.stringify(aclRoles)}</span>
                  </div>

                  <div className="field">
                    <label htmlFor="admin-token">X-Admin-Token</label>
                    <input
                      id="admin-token"
                      className="input mono"
                      value={token}
                      onChange={(event) => setToken(event.target.value)}
                      autoComplete="off"
                      spellCheck={false}
                    />
                  </div>

                  <button type="submit" className="btn btn-primary" disabled={uploading || !file}>
                    {uploading ? (
                      <>
                        <span className="spinner" /> 解析 + 切块 + 索引中…
                      </>
                    ) : (
                      "上传并索引"
                    )}
                  </button>

                  {uploading ? (
                    <div className="stream-hint">
                      <span className="spinner" /> 大文件需要几十秒，请勿关闭页面
                    </div>
                  ) : null}

                  {uploadError ? <ErrorState message={uploadError.detail} status={uploadError.status} /> : null}

                  {uploadResult ? (
                    <div className="banner info" style={{ flexDirection: "column", gap: 6 }}>
                      <div className="banner-title">上传成功 · {uploadResult.title}</div>
                      <div className="mono" style={{ fontSize: 11.5, lineHeight: 1.8 }}>
                        doc_id {uploadResult.document_id} · {uploadResult.pages} 页 · {uploadResult.paragraphs} 段 ·{" "}
                        {(uploadResult.elapsed_ms / 1000).toFixed(1)}s · {uploadResult.embedder}
                      </div>
                      <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                        {Object.entries(uploadResult.strategy_stats ?? {}).map(([name, stats]) => (
                          <span key={name} className="tag">
                            {name}: {String(stats.chunks ?? "?")} chunks
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </form>
              </div>

              <div className="card outline-card">
                <div className="card-head">
                  <span className="card-title">章节树 / outline</span>
                  <span className="card-sub">
                    {selected ? `${selected.title} · ${pageLabel(1, selected.pages)}` : "未选择文档"}
                  </span>
                </div>

                {!selected ? (
                  <EmptyState icon="◱" title="选择左侧文档" desc="点击列表中的可见文档，加载 structure 切块得到的章节树。" />
                ) : null}
                {selected && outlineLoading ? <LoadingState label="读取 outline…" rows={3} /> : null}
                {selected && outlineError ? (
                  <div style={{ padding: 14 }}>
                    <ErrorState message={outlineError.detail} status={outlineError.status} />
                  </div>
                ) : null}
                {selected && outline && outline.sections.length === 0 ? (
                  <EmptyState
                    icon="◌"
                    title="该文档没有章节信息"
                    desc={
                      outline.outline_source === "missing"
                        ? "该文档在章节树落库前入库（迁移前数据），重新入库后才会显示章节。"
                        : "解析后没有识别到章节标记（第 X 章 / 第 X 节 / markdown 标题）。"
                    }
                  />
                ) : null}
                {selected && outline && outline.sections.length > 0 ? (
                  <div className="outline-tree">
                    {outline.sections.map((section, index) => (
                      <button
                        key={`${section.title}-${index}`}
                        type="button"
                        className="outline-item"
                        style={{ width: "100%", background: "transparent", border: "0", borderLeft: "1px solid rgba(255,255,255,.09)", cursor: "pointer", textAlign: "left" }}
                        onClick={() =>
                          setViewer({
                            doc_id: selected.id,
                            doc_title: selected.title,
                            page_start: section.page,
                            page_end: section.page,
                            char_start: 0,
                            char_end: 0,
                            section_path: section.title,
                          })
                        }
                        title="打开该章节所在页"
                      >
                        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{section.title}</span>
                        <span className="outline-page">p.{section.page}</span>
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </div>
      </div>

      {viewer ? <PageViewer key={`${viewer.doc_id}-${viewer.page_start}`} target={viewer} onClose={() => setViewer(null)} /> : null}
    </>
  );
}