/**
 * 「读不到」的显式回显 —— 专门对付「把后端故障说成空态」这一类。
 *
 * 为什么单独一个组件：设置页的画像卡、云端模型、语音合成三块都是同一形态
 * （`load()` 里 catch 之后 state 留 null/[]，渲染分支把 null 当「还没配过」），
 * 于是同一句谎话在三处各写了一遍：后端一挂，界面就说「你还没有画像 / 尚未配置」。
 * 用户看到的后果是去多聊两句、去重填一遍表单，而不是去重试或看后端——方向完全错。
 *
 * 底色复用 global.css 的 `.set-warn` 原语（与知识库页的提示同一份），
 * 不新开一个「第 N 种提示条」。
 */
export default function LoadFail({
  what,
  error,
  onRetry,
}: {
  /** 读的是什么，例如「画像」「云端模型配置」 */
  what: string
  error: string
  onRetry: () => void
}) {
  return (
    <div className="set-warn">
      {what}读取失败：{error}。这不代表没配过 ——
      <button type="button" className="btn btn-ghost btn-sm" onClick={onRetry}>
        重试
      </button>
    </div>
  )
}
