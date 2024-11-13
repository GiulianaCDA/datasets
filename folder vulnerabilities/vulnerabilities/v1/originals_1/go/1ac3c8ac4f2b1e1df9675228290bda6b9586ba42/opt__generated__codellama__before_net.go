[PYTHON]
def handle_neighbors_packet(self, n: Node, pkt: IngressPacket) -> None:
    req = pkt.data  # type: ignore
    nodes = []
    for i, rn in enumerate(req.nodes):
        try:
            nn = self.intern_node_from_neighbours(pkt.remote_addr, rn)
        except ValidationError as err:
            log.debug(f"invalid neighbour ({rn}) from {n.id[:8]}@{pkt.remote_addr}: {err}")
            continue
        nodes.append(nn)
        # Start validation of query results immediately.
        # This fills the table quickly.
        # TODO: generates way too many packets, maybe do it via queue.
        if nn.state == NodeState.unknown:
            self.transition(nn, NodeState.verify_init)
    # TODO: don't ignore second packet
    n.pending_neighbours.reply <- nodes
    n.pending_neighbours = None
    # Now that this query is done, start the next one.
    n.start_next_query(self)
[/PYTHON]
