"use client";

/**
 * 把答案正文里的 [n] 渲染成可点击引用标注：点击后由父组件滚动并高亮对应证据卡。
 */
export default function AnswerText({
  text,
  knownRefs,
  onCite,
}: {
  text: string;
  knownRefs: Set<number>;
  onCite: (ref: number) => void;
}) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <>
      {parts.map((part, index) => {
        const match = /^\[(\d+)\]$/.exec(part);
        if (!match || !match[1]) return <span key={index}>{part}</span>;
        const ref = Number(match[1]);
        const known = knownRefs.has(ref);
        return (
          <button
            key={index}
            type="button"
            className={`cite-ref${known ? "" : " missing"}`}
            title={known ? `跳转到证据 [${ref}]` : `引用 [${ref}] 无对应证据（后端未返回该条）`}
            aria-label={`引用 ${ref}`}
            onClick={() => {
              if (known) onCite(ref);
            }}
          >
            {ref}
          </button>
        );
      })}
    </>
  );
}