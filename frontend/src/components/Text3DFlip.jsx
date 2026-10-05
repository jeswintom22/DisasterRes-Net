"use client"

import React, { memo, useCallback, useEffect, useMemo, useRef } from "react"
import { useAnimate } from "motion/react"

import { cn } from "@/lib/utils"

const HAS_SEGMENTER = typeof Intl !== "undefined" && "Segmenter" in Intl

const splitIntoCharacters = (text) => {
  if (HAS_SEGMENTER) {
    const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" })
    return Array.from(segmenter.segment(text), ({ segment }) => segment)
  }
  return Array.from(text)
}

const extractTextFromChildren = (children) => {
  if (children == null) return ""
  if (typeof children === "string") return children
  if (typeof children === "number") return String(children)

  if (Array.isArray(children)) {
    return children.map(extractTextFromChildren).join("")
  }

  if (React.isValidElement(children)) {
    const childText = children.props.children
    if (childText != null) {
      return extractTextFromChildren(childText)
    }
  }

  return ""
}

const ROTATION_MAP = {
  top: "rotateX(90deg)",
  right: "rotateY(90deg)",
  bottom: "rotateX(-90deg)",
  left: "rotateY(-90deg)",
}

const DEFAULT_TRANSITION = {
  type: "spring",
  damping: 30,
  stiffness: 300,
}

const Text3DFlip = ({
  children,
  as: ElementTag = "p",
  className,
  textClassName,
  flipTextClassName,
  staggerDuration = 0.05,
  staggerFrom = "first",
  transition = DEFAULT_TRANSITION,
  rotateDirection = "right",
  ...props
}) => {
  const isAnimatingRef = useRef(false)
  const isMountedRef = useRef(false)
  const [scope, animate] = useAnimate()

  const rotationTransform = ROTATION_MAP[rotateDirection]

  useEffect(() => {
    isMountedRef.current = true

    return () => {
      isMountedRef.current = false
      isAnimatingRef.current = false
    }
  }, [])

  const text = useMemo(() => {
    try {
      return extractTextFromChildren(children)
    } catch {
      return ""
    }
  }, [children])

  const characters = useMemo(() => {
    const words = text.split(" ")
    return words.map((word, i) => ({
      characters: splitIntoCharacters(word),
      needsSpace: i !== words.length - 1,
    }))
  }, [text])

  const charOffsets = useMemo(() => {
    const offsets = [0]
    for (const word of characters) {
      offsets.push(offsets.at(-1) + word.characters.length)
    }
    return offsets
  }, [characters])

  const getStaggerDelay = useCallback(
    (index, totalChars) => {
      if (staggerFrom === "first") return index * staggerDuration
      if (staggerFrom === "last")
        return (totalChars - 1 - index) * staggerDuration
      if (staggerFrom === "center") {
        const center = Math.floor(totalChars / 2)
        return Math.abs(center - index) * staggerDuration
      }
      if (staggerFrom === "random") {
        const randomIndex = Math.floor(Math.random() * totalChars)
        return Math.abs(randomIndex - index) * staggerDuration
      }
      return Math.abs(staggerFrom - index) * staggerDuration
    },
    [staggerFrom, staggerDuration]
  )

  const handleHoverStart = useCallback(async () => {
    if (isAnimatingRef.current) return
    isAnimatingRef.current = true

    try {
      const totalChars = characters.reduce(
        (sum, word) => sum + word.characters.length,
        0
      )

      const delays = Array.from({ length: totalChars }, (_, i) =>
        getStaggerDelay(i, totalChars)
      )

      await animate(
        ".text-3d-flip-char",
        { transform: rotationTransform },
        {
          ...transition,
          delay: (i) => delays[i],
        }
      )

      if (!isMountedRef.current) return

      await animate(
        ".text-3d-flip-char",
        { transform: "rotateX(0deg) rotateY(0deg)" },
        { duration: 0 }
      )
    } finally {
      if (isMountedRef.current) {
        isAnimatingRef.current = false
      }
    }
  }, [characters, transition, getStaggerDelay, rotationTransform, animate])

  return (
    <ElementTag
      className={cn("relative flex flex-wrap", className)}
      onMouseEnter={handleHoverStart}
      ref={scope}
      {...props}
    >
      <span className="sr-only">{text}</span>

      {characters.map((wordObj, wordIndex) => (
        <span key={wordIndex} className="inline-flex">
          {wordObj.characters.map((char, charIndex) => (
            <CharBox
              key={charOffsets[wordIndex] + charIndex}
              char={char}
              textClassName={textClassName}
              flipTextClassName={flipTextClassName}
              rotateDirection={rotateDirection}
            />
          ))}
          {wordObj.needsSpace && <span className="whitespace-pre"> </span>}
        </span>
      ))}
    </ElementTag>
  )
}

const SECOND_FACE_TRANSFORMS = {
  top: "rotateX(-90deg) translateZ(0.5lh)",
  right:
    "rotateY(90deg) translateX(50%) rotateY(-90deg) translateX(-50%) rotateY(-90deg) translateX(50%)",
  bottom: "rotateX(90deg) translateZ(0.5lh)",
  left: "rotateY(90deg) translateX(50%) rotateY(-90deg) translateX(50%) rotateY(-90deg) translateX(50%)",
}

const FRONT_FACE_TRANSFORMS = {
  top: "translateZ(0.5lh)",
  bottom: "translateZ(0.5lh)",
  left: "rotateY(90deg) translateX(50%) rotateY(-90deg)",
  right: "rotateY(-90deg) translateX(50%) rotateY(90deg)",
}

const CONTAINER_TRANSFORMS = {
  top: "translateZ(-0.5lh)",
  bottom: "translateZ(-0.5lh)",
  left: "rotateY(90deg) translateX(50%) rotateY(-90deg)",
  right: "rotateY(90deg) translateX(50%) rotateY(-90deg)",
}

const CharBox = memo(
  ({ char, textClassName, flipTextClassName, rotateDirection }) => (
    <span
      className="text-3d-flip-char inline transform-3d"
      style={{ transform: CONTAINER_TRANSFORMS[rotateDirection] }}
    >
      <span
        className={cn("relative h-[1lh] backface-hidden", textClassName)}
        style={{ transform: FRONT_FACE_TRANSFORMS[rotateDirection] }}
      >
        {char}
      </span>
      <span
        className={cn(
          "absolute top-0 left-0 h-[1lh] backface-hidden",
          flipTextClassName
        )}
        style={{ transform: SECOND_FACE_TRANSFORMS[rotateDirection] }}
      >
        {char}
      </span>
    </span>
  )
)

CharBox.displayName = "CharBox"
Text3DFlip.displayName = "Text3DFlip"

export { Text3DFlip }
export default Text3DFlip
