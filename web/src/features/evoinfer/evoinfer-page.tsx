import type { ReactElement } from "react";
import {
  BrainCircuit,
  CheckCircle2,
  GitBranch,
  Gauge,
  Layers3,
  Lightbulb,
  Network,
  Repeat2,
} from "lucide-react";

const PIPELINE_STEPS = [
  {
    icon: Gauge,
    title: "真实任务",
    text: "agent 在不同硬件、模型架构和推理后端上完成大量推理优化 task，所有结论来自可复现 benchmark。",
  },
  {
    icon: BrainCircuit,
    title: "经验抽取",
    text: "每次优化后，把环境、baseline、瓶颈、改动、收益和失败原因沉淀到记忆库，而不是只保留聊天记录。",
  },
  {
    icon: Network,
    title: "关联成网",
    text: "记忆会按硬件、模型类型、算子、后端、精度和性能指标建立关联，逐步形成可迁移的工程知识网络。",
  },
  {
    icon: Lightbulb,
    title: "创新迁移",
    text: "当经验数量足够多，agent 能在新任务中看到相似结构，发现可以做的优化创新点，而不是只执行局部指令。",
  },
] as const;

const CAPABILITIES = [
  "从 ncu、nsys、benchmark 和运行日志中判断真正瓶颈",
  "在 CUDA、Triton、PyTorch、vLLM、TensorRT 等后端之间比较取舍",
  "针对单个算子、KV cache、batching、并发、精度和内存布局提出优化路线",
  "把成功经验复用到相似任务，把失败经验转化为排查约束",
] as const;

export function EvoInferPage(): ReactElement {
  return (
    <main className="h-full w-full overflow-y-auto bg-background">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-10 px-6 py-8 lg:px-10">
        <section className="space-y-6 border-b border-border pb-8">
          <div className="space-y-5">
            <div className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1 text-xs font-medium text-muted-foreground">
              <Repeat2 className="size-3.5" />
              Self-evolving inference optimization agent
            </div>
            <div className="space-y-3">
              <h1 className="text-3xl font-semibold tracking-normal text-foreground md:text-4xl">
                EvoInfer
              </h1>
              <p className="max-w-3xl text-base leading-7 text-muted-foreground">
                EvoInfer 的目标是通过自进化的方式，让 code agent 在推理优化上达到接近人类工程师的工程全局理解程度。
                它不是只记住一次对话，而是在持续完成优化任务的过程中，把可验证经验写入记忆库，逐步形成面向推理系统的长期工程判断。
              </p>
            </div>
          </div>

          <figure className="mx-auto w-full max-w-5xl overflow-hidden rounded-lg border border-border bg-muted/25">
            <img
              alt="EvoInfer 自进化推理优化原理图"
              className="aspect-[16/9] w-full object-cover"
              height={941}
              src="/evoinfer-principle.png"
              width={1672}
            />
          </figure>

          <div className="mx-auto grid max-w-5xl gap-3 rounded-lg border border-border bg-muted/25 p-4">
            <div className="flex items-center gap-2 text-sm font-medium text-foreground">
              <Layers3 className="size-4 text-muted-foreground" />
              核心假设
            </div>
            <p className="text-sm leading-6 text-muted-foreground">
              当 agent 完成足够多真实推理优化 task，并把任务轨迹、环境约束、性能指标和优化结果组织成可检索的记忆网络后，它能获得跨任务的全局理解，
              从而主动发现新的优化创新点。
            </p>
          </div>
        </section>

        <section className="space-y-4">
          <div className="flex items-center gap-2">
            <GitBranch className="size-4 text-muted-foreground" />
            <h2 className="text-lg font-semibold text-foreground">自进化闭环</h2>
          </div>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            {PIPELINE_STEPS.map((step) => {
              const Icon = step.icon;
              return (
                <article
                  className="rounded-lg border border-border bg-background p-4"
                  key={step.title}
                >
                  <div className="mb-3 flex size-8 items-center justify-center rounded-md bg-secondary text-muted-foreground">
                    <Icon className="size-4" />
                  </div>
                  <h3 className="text-sm font-semibold text-foreground">
                    {step.title}
                  </h3>
                  <p className="mt-2 text-sm leading-6 text-muted-foreground">
                    {step.text}
                  </p>
                </article>
              );
            })}
          </div>
        </section>

        <section className="grid gap-8 border-t border-border pt-8 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="space-y-3">
            <h2 className="text-lg font-semibold text-foreground">
              为什么记忆库会改变 agent 的能力边界
            </h2>
            <p className="text-sm leading-7 text-muted-foreground">
              单次 session 里的 agent 只能围绕当前上下文推理；EvoInfer 让它把跨项目经验长期保留下来。随着经验数量增加，
              它会从“完成一个优化需求”转向“理解整个推理系统的约束、瓶颈和可迁移机会”。
            </p>
          </div>

          <div className="grid gap-2">
            {CAPABILITIES.map((item) => (
              <div
                className="flex items-start gap-3 rounded-md border border-border bg-muted/20 px-3 py-2.5 text-sm text-muted-foreground"
                key={item}
              >
                <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-600" />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
