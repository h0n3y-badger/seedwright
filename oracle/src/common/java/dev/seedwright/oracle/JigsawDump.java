package dev.seedwright.oracle;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import net.minecraft.core.Holder;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraft.world.level.chunk.ChunkGeneratorStructureState;
import net.minecraft.world.level.chunk.status.ChunkStatus;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.structure.BoundingBox;
import net.minecraft.world.level.levelgen.structure.PoolElementStructurePiece;
import net.minecraft.world.level.levelgen.structure.Structure;
import net.minecraft.world.level.levelgen.structure.StructurePiece;
import net.minecraft.world.level.levelgen.structure.StructureSet;
import net.minecraft.world.level.levelgen.structure.StructureStart;

/**
 * Jigsaw layouts for seedwright's jigsaw.h: every start of the village / pillager outpost sets within
 * -Dseedwright.oracle.villages=R chunks, with its pieces (element, rotation, bounding box) and the terrain heights
 * (getBaseHeight WORLD_SURFACE_WG) of every column covered by a piece. Written to villages-&lt;seed&gt;.txt.
 * Uses only API that is the same in 1.21.1 - 26.3 (shared by both oracle builds).
 */
public final class JigsawDump
{
	private static final Pattern LEFT = Pattern.compile("Left\\[([^\\]]+)\\]");
	private static final Pattern FEATURE = Pattern.compile("/ ([a-z0-9_.:/-]+)\\]");

	private JigsawDump() {}

	/** element toString -> template id | feature:id | list:first | empty */
	static String describe(String s)
	{
		if (s.startsWith("List["))
		{
			return "list:" + describe(s.substring(5));
		}
		if (s.startsWith("Feature["))
		{
			Matcher m = FEATURE.matcher(s);
			return m.find() ? "feature:" + m.group(1) : "feature:?";
		}
		if (s.startsWith("Empty"))
		{
			return "empty";
		}
		Matcher m = LEFT.matcher(s);
		return m.find() ? m.group(1) : "?" + s.replace(' ', '_');
	}

	public static void dump(ServerLevel level, ChunkGenerator generator, RandomState randomState, ChunkGeneratorStructureState state, long seed,
		int radius, Function<Holder<StructureSet>, String> setName, Map<Structure, String> structureNames) throws Exception
	{
		StringBuilder out = new StringBuilder("seed ").append(seed).append('\n');
		int count = 0;
		for (Holder<StructureSet> set : state.possibleStructureSets())
		{
			String sn = setName.apply(set);
			if (!sn.endsWith(":villages") && !sn.endsWith(":pillager_outposts"))
			{
				continue;
			}
			for (int x = -radius; x <= radius; x++)
			{
				for (int z = -radius; z <= radius; z++)
				{
					if (!set.value().placement().isStructureChunk(state, x, z))
					{
						continue;
					}
					ChunkAccess chunk = level.getChunk(x, z, ChunkStatus.STRUCTURE_STARTS, true);
					for (Map.Entry<Structure, StructureStart> e : chunk.getAllStarts().entrySet())
					{
						StructureStart start = e.getValue();
						boolean inSet = false;
						for (StructureSet.StructureSelectionEntry se : set.value().structures())
						{
							inSet |= se.structure().value() == e.getKey();
						}
						if (!inSet || !start.isValid())
						{
							continue;
						}
						String name = structureNames.getOrDefault(e.getKey(), "?");
						out.append("structure ").append(name).append(' ').append(x).append(' ').append(z).append(' ').append(start.getPieces().size()).append('\n');
						int x0 = Integer.MAX_VALUE, z0 = Integer.MAX_VALUE, x1 = Integer.MIN_VALUE, z1 = Integer.MIN_VALUE;
						for (StructurePiece p : start.getPieces())
						{
							BoundingBox b = p.getBoundingBox();
							String desc = "?", rot = "?";
							if (p instanceof PoolElementStructurePiece pe)
							{
								desc = describe(pe.getElement().toString());
								rot = Integer.toString(pe.getRotation().ordinal());
							}
							out.append(desc).append(' ').append(rot).append(' ').append(b.minX()).append(' ').append(b.minY()).append(' ').append(b.minZ())
								.append(' ').append(b.maxX()).append(' ').append(b.maxY()).append(' ').append(b.maxZ()).append('\n');
							x0 = Math.min(x0, b.minX());
							z0 = Math.min(z0, b.minZ());
							x1 = Math.max(x1, b.maxX());
							z1 = Math.max(z1, b.maxZ());
						}
						// heights of every column inside some piece ('-' elsewhere): all vanilla height queries are there
						int nx = x1 - x0 + 1, nz = z1 - z0 + 1;
						boolean[] need = new boolean[nx * nz];
						for (StructurePiece p : start.getPieces())
						{
							BoundingBox b = p.getBoundingBox();
							for (int bz = b.minZ(); bz <= b.maxZ(); bz++)
							{
								for (int bx = b.minX(); bx <= b.maxX(); bx++)
								{
									need[(bz - z0) * nx + bx - x0] = true;
								}
							}
						}
						out.append("heights ").append(x0).append(' ').append(z0).append(' ').append(nx).append(' ').append(nz).append('\n');
						for (int k = 0; k < nz; k++)
						{
							for (int i = 0; i < nx; i++)
							{
								if (i > 0)
								{
									out.append(' ');
								}
								if (need[k * nx + i])
								{
									out.append(generator.getBaseHeight(x0 + i, z0 + k, Heightmap.Types.WORLD_SURFACE_WG, level, randomState));
								}
								else
								{
									out.append('-');
								}
							}
							out.append('\n');
						}
						count++;
					}
				}
			}
		}
		out.append("end ").append(count).append('\n');
		Files.writeString(Path.of("villages-" + seed + ".txt"), out);
	}
}
